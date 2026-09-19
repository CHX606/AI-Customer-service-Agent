# Apply only deployment-script fixes to the already built, verified image.
# Tag its recorded original digest as :packaged before invoking this build.
FROM ai-customer-service-backend:packaged
USER 0
# The original requirements omitted Docx2txtLoader's optional runtime dependency.
RUN python -m pip install --no-deps docx2txt==0.9 && python -m pip check
USER 10001:10001
COPY --chown=10001:10001 requirements.txt /app/requirements.txt
COPY --chown=10001:10001 infra/deploy /app/infra/deploy
COPY --chown=10001:10001 scripts/export_reranker_onnx.py /app/scripts/export_reranker_onnx.py
COPY --chown=10001:10001 back/bootstrap.py /app/back/bootstrap.py
COPY --chown=10001:10001 back/tenant/service.py /app/back/tenant/service.py
COPY --chown=10001:10001 back/core/features.py /app/back/core/features.py
COPY --chown=10001:10001 back/interfaces/http/app.py back/interfaces/http/chat.py /app/back/interfaces/http/
COPY --chown=10001:10001 back/knowledge/images/cpu_attention.py back/knowledge/images/parser.py back/knowledge/images/query.py back/knowledge/images/semantics.py /app/back/knowledge/images/
COPY --chown=10001:10001 back/knowledge/ingestion/knowledge_loader.py /app/back/knowledge/ingestion/knowledge_loader.py
COPY --chown=10001:10001 back/knowledge/indexing/service.py /app/back/knowledge/indexing/service.py
