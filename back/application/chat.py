"""聊天应用服务：统一普通、图片和流式聊天的会话及缓存行为。"""
import logging
import hashlib
from copy import deepcopy
from dataclasses import asdict
from langchain_core.messages import AIMessage, HumanMessage
from back.application.ports import AnswerCache, ChatEngine, SessionRepository, TenantReader
from back.domain.chat import CacheHit as SemanticCacheHit, ChatCommand, ChatResult, ChatState, PreparedChat
from back.domain.errors import ApplicationError, Conflict, DependencyUnavailable, InvalidRequest, NotFound, ProcessingFailed
from back.domain.tenant import validate_tenant_id

CustomerServiceState = ChatState
logger = logging.getLogger(__name__)

def _first_turn_question(prepared: PreparedChat) -> str | None:
    """仅允许可独立理解的首轮纯文本问题使用语义缓存。"""
    messages = prepared.graph_input.get("messages", [])
    if len(messages) != 1 or not isinstance(messages[0], HumanMessage):
        return None
    content = messages[0].content
    return content.strip() if isinstance(content, str) and content.strip() else None

def _cache_hit_state(
    prepared: PreparedChat,
    hit: SemanticCacheHit,
) -> CustomerServiceState:
    """把缓存答案转换成普通会话状态，以便后续追问仍有上下文。"""
    question = _first_turn_question(prepared) or "已缓存问题"
    return {
        **prepared.graph_input,
        "messages": [
            *prepared.graph_input.get("messages", []),
            AIMessage(content=hit.answer),
        ],
        "active_issue": {
            "summary": question,
            "status": "answered",
        },
        "scope": "in_scope",
        "action": "retrieve",
        "route_source": "semantic_cache",
        "evidence_status": "sufficient",
    }

def _should_store_semantic_answer(result: CustomerServiceState) -> bool:
    """只缓存有可靠依据的知识库回答或企业资料回答。"""
    action = result.get("action")
    if action == "profile":
        return result.get("scope") == "in_scope"
    return action == "retrieve" and result.get("evidence_status") == "sufficient"


class ChatService:
    def __init__(self, engine: ChatEngine, sessions: SessionRepository, cache: AnswerCache, tenants: TenantReader):
        self.engine = engine
        self.sessions = sessions
        self.cache = cache
        self.tenants = tenants

    def prepare(self, command: ChatCommand) -> PreparedChat:
        try:
            tenant_id = validate_tenant_id(command.tenant_id)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
        if self.tenants.get(tenant_id) is None:
            raise NotFound(f"未找到租户 '{tenant_id}' 的配置资料")
        key = (tenant_id, command.session_id)
        snapshot = self.sessions.load(key)
        messages = list(snapshot.state.get("messages", []))
        active_issue = snapshot.state.get("active_issue")
        target = next((i for i, message in enumerate(messages)
                       if command.turn_id and isinstance(message, HumanMessage) and message.id == command.turn_id), None)
        if target is None and command.regenerate and command.previous_answer_hash:
            # 兼容升级前没有逐轮 ID 的会话；只有答案唯一匹配才允许回退。
            matches = [i - 1 for i, message in enumerate(messages) if i > 0
                       and isinstance(message, AIMessage) and isinstance(messages[i - 1], HumanMessage)
                       and isinstance(message.content, str)
                       and hashlib.sha256(message.content.encode()).hexdigest() == command.previous_answer_hash]
            if len(matches) == 1:
                target = matches[0]
        if command.regenerate and target is None:
            raise Conflict("这条回复的原始上下文已不可用，请重新提问。")
        regenerating = target is not None
        if regenerating:
            original = messages[target]
            active_issue = original.additional_kwargs.get("active_issue_before")
            question = original.content  # 图片重生复用已保存的识别结果。
            messages = messages[:target]
        else:
            question = command.message
        human = HumanMessage(content=question, id=command.turn_id,
                             additional_kwargs={"active_issue_before": deepcopy(active_issue)})
        try:
            cache_revision = self.cache.revision(tenant_id) if not messages else None
        except Exception:
            logger.exception("缓存版本读取失败，本次请求跳过答案缓存")
            cache_revision = None
        return PreparedChat(tenant_id, command.session_id, key, {
            "tenant_id": tenant_id,
            "messages": [*messages[-40:], human],
            "active_issue": active_issue,
        }, snapshot.version, cache_revision, regenerating)

    def find_cached(self, prepared: PreparedChat) -> SemanticCacheHit | None:
        """缓存不可用时安全降级到正常 Agent，不影响客服请求。"""
        question = _first_turn_question(prepared)
        if question is None or prepared.regenerate or prepared.cache_revision is None:
            return None
        try:
            hit = self.cache.find(prepared.tenant_id, question)
            if self.cache.revision(prepared.tenant_id) != prepared.cache_revision:
                return None
        except Exception:
            logger.exception("语义问答缓存查询失败，将继续正常客服流程")
            return None
        if hit is not None:
            logger.info(
                "语义问答缓存命中 tenant=%s exact=%s similarity=%.4f",
                prepared.tenant_id,
                hit.exact,
                hit.similarity,
            )
        return hit

    def finalize(self, prepared: PreparedChat, result: ChatState, *, allow_cache_store: bool = True) -> ChatResult:
        messages = result.get("messages", [])
        if not messages or not isinstance(messages[-1].content, str):
            raise ProcessingFailed("客服流程未返回有效回答。")
        final_answer = messages[-1].content
        self.sessions.save(prepared.session_key, result, prepared.version)
        question = _first_turn_question(prepared)
        if allow_cache_store and prepared.cache_revision is not None and question is not None and _should_store_semantic_answer(result):
            try:
                self.cache.store(prepared.tenant_id, question, final_answer, expected_revision=prepared.cache_revision)
            except Exception:
                logger.exception("语义缓存写入失败，本次回答继续返回")
        return ChatResult(final_answer, prepared.session_id, prepared.tenant_id)

    def execute(self, command: ChatCommand, *, allow_semantic_cache: bool = True) -> ChatResult:
        prepared = self.prepare(command)
        if allow_semantic_cache:
            hit = self.find_cached(prepared)
            if hit is not None:
                return self.finalize(prepared, _cache_hit_state(prepared, hit), allow_cache_store=False)
        try:
            result = self.engine.invoke(prepared.graph_input)
        except ApplicationError:
            raise
        except Exception as exc:
            logger.exception("聊天引擎调用失败 tenant=%s", prepared.tenant_id)
            raise DependencyUnavailable("客服服务暂不可用，请稍后重试。") from exc
        return self.finalize(prepared, result)

    def stream(self, prepared: PreparedChat):
        yield {"type": "status", "message": "正在理解您的问题…"}
        try:
            hit = self.find_cached(prepared)
            if hit is not None:
                result = self.finalize(prepared, _cache_hit_state(prepared, hit), allow_cache_store=False)
                yield {"type": "final", **asdict(result)}
                return
            final_state = None
            for event in self.engine.stream(prepared.graph_input):
                if event["type"] == "state":
                    final_state = event["state"]
                else:
                    yield event
            if final_state is None:
                raise ProcessingFailed("客服流程未返回结果")
            result = self.finalize(prepared, final_state)
            yield {"type": "final", **asdict(result)}
        except Exception:
            logger.exception("流式聊天处理失败 tenant=%s", prepared.tenant_id)
            yield {"type": "error", "message": "客服请求处理失败，请稍后重试。"}
