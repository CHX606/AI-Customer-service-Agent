// Isolated UI acceptance backend. Data lives only in this process; no real services are called.
// Run: node scripts/ui-fixture.mjs
// Start a separate Vite instance with VITE_API_BASE_URL=http://127.0.0.1:8011 on port 5174.
import { createServer } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';

const timestamp = '2026-09-11T02:00:00Z';
let profile = {
  tenant_id: 'default', company_name: '可乐云', brand_name_en: 'KeleCloud',
  assistant_name: '可乐云 AI 智能客服', short_description: '用于界面验收的测试企业资料。',
  business_scope: ['账号与订单咨询', '客户端问题排查'], business_hours: '每天 09:00–22:00',
  public_contact: '在线客服工单', welcome_title: '你好，我是可乐云智能客服',
  welcome_description: '随时向我咨询业务规则、账号异常排查，支持直接粘贴或上传故障截图进行智能分析。',
  tone: '简洁、友好、专业', handoff_message: '请联系人工客服进一步处理。',
  suggested_questions: ['续费之后为什么流量没有重置？', '忘记账号或密码应该怎么办？', '软件突然不能使用了，应该如何排查？', '如何更新订阅或重新导入节点？'],
  created_at: timestamp, updated_at: timestamp,
};
const makeFile = (id, name, status = 'ready') => ({
  source_id: id, tenant_id: 'default', original_filename: name, stored_filename: name,
  file_type: name.split('.').at(-1), content_hash: id, status,
  chunk_count: 12, created_at: timestamp, updated_at: timestamp,
  error_message: status === 'failed' ? '测试文档格式无法解析，请检查后重新上传。' : null,
});
let files = [makeFile('fixture-guide', '客服使用指南.pdf'), makeFile('fixture-error', '套餐与退款规则说明（界面测试）.docx', 'failed')];
const requests = [];
const answer = '这是隔离测试服务的示例回复。\n\n请先确认订单状态，再查看套餐的流量重置时间。续费与流量重置可能采用不同的规则。\n\n如果仍有疑问，请提供订单编号并联系人工客服。';

createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', 'http://127.0.0.1:5174');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Tenant-Token');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  const url = new URL(req.url, 'http://127.0.0.1:8011');
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString();
  const jsonBody = req.headers['content-type']?.includes('application/json') && raw ? JSON.parse(raw) : null;
  const field = (name) => raw.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r\\n]*)`))?.[1];
  const reply = (value, status = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(value));
  };
  if (url.pathname === '/__requests') { reply(requests); return; }
  if (req.method !== 'GET') requests.push({ method: req.method, path: url.pathname, body: jsonBody, image: raw.includes('name="image"'), message: field('message'), session_id: field('session_id') });
  if (url.pathname === '/health') { reply({ status: 'ok' }); return; }
  if (url.pathname === '/public/profile' || url.pathname === '/admin/profile') {
    if (req.method === 'PUT') profile = { ...profile, ...jsonBody };
    reply(profile); return;
  }
  if (url.pathname === '/admin/knowledge/files') {
    if (req.method === 'POST') {
      const file = makeFile(`fixture-upload-${files.length}`, raw.match(/filename="([^"]+)"/)?.[1] || '测试文档.txt');
      files.push(file); reply(file); return;
    }
    reply(files); return;
  }
  if (url.pathname.startsWith('/admin/knowledge/files/')) {
    const id = url.pathname.split('/')[4];
    if (id === 'fixture-error') { reply({ detail: '测试失败提示：请稍后重试。' }, 500); return; }
    if (req.method === 'DELETE') { files = files.filter((file) => file.source_id !== id); reply({ ok: true }); return; }
    reply(files.find((file) => file.source_id === id)); return;
  }
  if (url.pathname === '/chat/stream') {
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    const emit = (event) => res.write(`${JSON.stringify(event)}\n`);
    try {
      emit({ type: 'status', message: '正在查阅企业资料…' });
      for (const delta of answer.match(/.{1,12}|\n/g)) {
        await delay(jsonBody.message.includes('慢速') ? 400 : 20);
        if (res.destroyed) return;
        emit({ type: 'token', delta });
      }
      emit({ type: 'final', answer, session_id: jsonBody.session_id, tenant_id: 'default' });
      res.end();
    } catch { res.end(); }
    return;
  }
  if (url.pathname === '/chat/image') { reply({ answer: '已收到测试截图，图片提问链路正常。', session_id: field('session_id'), tenant_id: 'default' }); return; }
  reply({ detail: 'UI fixture endpoint not found' }, 404);
}).listen(8011, '127.0.0.1', () => console.log('UI fixture: http://127.0.0.1:8011 (memory only)'));
