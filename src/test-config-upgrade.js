// 升级容错：旧 config.json 的 LLM 块不可用时，不能连带把 security / temperature 等
// 兄弟字段一起重置（升级后最常见的"配置全没了"根因）。
//
// 隔离策略：把 BAILONGMA_USER_DIR 指向临时目录，paths.configFile 随之落在临时目录里。
// 每个场景重写同一个 config.json，再用带版本号的 URL 重新 import config.js（绕过模块缓存，
// 让顶层加载逻辑对新文件重跑一遍；paths.js 已缓存，configFile 路径保持不变）。
//
// Run: node src/test-config-upgrade.js

import fs from 'fs'
import os from 'os'
import path from 'path'

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'blm-config-'))
process.env.BAILONGMA_USER_DIR = tmp

// 清掉可能存在的 LLM 环境变量，否则 LLM 块不可用时会走 env 兜底而误判为已激活
for (const k of [
  'DEEPSEEK_API_KEY', 'MINIMAX_API_KEY', 'OPENAI_API_KEY', 'DASHSCOPE_API_KEY',
  'MOONSHOT_API_KEY', 'ZHIPU_API_KEY', 'MIMO_API_KEY',
]) delete process.env[k]

const configFile = path.join(tmp, 'config.json')

let failed = 0
function assert(cond, label) {
  if (!cond) { console.error(`FAIL: ${label}`); failed += 1; process.exitCode = 1 }
  else console.log(`PASS: ${label}`)
}

let v = 0
async function loadFresh(json) {
  fs.writeFileSync(configFile, JSON.stringify(json, null, 2), 'utf-8')
  v += 1
  return await import(`./config.js?v=${v}`)
}

// ── 场景 A：provider 是新版不认识的名字（模拟改名/删除的旧 provider）──
{
  const { config } = await loadFresh({
    provider: 'some-removed-provider',
    apiKey: 'sk-whatever-old-key-1234567890',
    model: 'old-model',
    temperature: 1.3,
    security: { fileSandbox: false, execSandbox: false, blockedTools: ['exec_command'] },
    voice: { voiceProvider: 'aliyun', aliyunApiKey: 'sk-aliyunkeyplaceholder1234567890' },
  })
  assert(config.needsActivation === true, 'A: 未知 provider → LLM 标记为待激活')
  assert(config.provider === null, 'A: 未知 provider 不被错误激活')
  // 关键：兄弟字段必须存活，不能被一起重置回默认
  assert(config.temperature === 1.3, 'A: temperature 在 LLM 不可用时仍被保留')
  assert(config.security.execSandbox === false, 'A: execSandbox=false 被保留（不会悄悄重新开启沙盒）')
  assert(config.security.fileSandbox === false, 'A: fileSandbox=false 被保留')
  assert(JSON.stringify(config.security.blockedTools) === JSON.stringify(['exec_command']), 'A: blockedTools 被保留')
  // voice 块由 getVoiceConfig 直接读盘，文件未被改写即存活
  const vc = JSON.parse(fs.readFileSync(configFile, 'utf-8')).voice
  assert(vc && vc.aliyunApiKey, 'A: voice 块原样留在文件里未被擦除')
}

// ── 场景 B：合法 provider，但 model 在新版列表里已不存在 → 激活成功且回退默认，security 仍保留 ──
{
  const { config, __internals } = await loadFresh({
    provider: 'deepseek',
    apiKey: 'sk-deepseek-valid-key-1234567890',
    model: 'deepseek-some-retired-model',
    temperature: 0.9,
    security: { execSandbox: false },
  })
  const validIds = new Set(__internals.DEEPSEEK_MODELS.map(m => m.id))
  assert(config.needsActivation === false, 'B: 合法 provider 正常激活')
  assert(config.provider === 'deepseek', 'B: provider 正确')
  assert(validIds.has(config.model), `B: 退役 model 已归一到 deepseek 合法值（实得 ${config.model}）`)
  assert(config.security.execSandbox === false, 'B: 激活路径下 security 同样保留')
}

// ── 场景 C：custom provider（baseURL + model 齐全）正常激活 ──
{
  const { config } = await loadFresh({
    provider: 'custom',
    apiKey: 'none',
    model: 'my-local-model',
    baseURL: 'http://127.0.0.1:1234/v1',
  })
  assert(config.needsActivation === false, 'C: custom provider 正常激活')
  assert(config.provider === 'custom' && config.model === 'my-local-model', 'C: custom model 原样保留（不归一）')
  assert(config.baseURL === 'http://127.0.0.1:1234/v1', 'C: custom baseURL 保留')
}

// ── 场景 D：损坏的 JSON → 整体回落未激活，不抛异常 ──
{
  fs.writeFileSync(configFile, '{ this is not valid json', 'utf-8')
  v += 1
  const { config } = await import(`./config.js?v=${v}`)
  assert(config.needsActivation === true, 'D: 损坏文件 → 未激活且不崩溃')
}

// ── 场景 E：schema 迁移 v0 → v1，旧版 seedance 块拆到独立 seedance.json ──
{
  const seedanceFile = path.join(tmp, 'seedance.json')
  try { fs.rmSync(seedanceFile, { force: true }) } catch {}
  const { config } = await loadFresh({
    provider: 'deepseek',
    apiKey: 'sk-deepseek-valid-key-1234567890',
    model: 'deepseek-v4-pro',
    seedance: { apiKey: 'ark-legacy-key', model: 'doubao-seedance-x', baseURL: 'https://ark.example/v3' },
    voice: { voiceProvider: 'aliyun' },
  })
  assert(config.needsActivation === false, 'E: 迁移后 LLM 仍正常激活')
  const after = JSON.parse(fs.readFileSync(configFile, 'utf-8'))
  assert(after.schemaVersion === 1, 'E: config.json 被打上 schemaVersion=1')
  assert(after.seedance === undefined, 'E: seedance 块已从 config.json 移除')
  assert(after.voice && after.voice.voiceProvider === 'aliyun', 'E: 其它块（voice）在迁移中保留')
  assert(fs.existsSync(seedanceFile), 'E: seedance.json 独立文件已生成')
  const sd = JSON.parse(fs.readFileSync(seedanceFile, 'utf-8'))
  assert(sd.apiKey === 'ark-legacy-key' && sd.model === 'doubao-seedance-x', 'E: seedance 数据完整搬迁')
}

// ── 场景 F：已是最新 schemaVersion 的文件不被重复迁移 / 改写 ──
{
  await loadFresh({
    schemaVersion: 1,
    provider: 'deepseek',
    apiKey: 'sk-deepseek-valid-key-1234567890',
    model: 'deepseek-v4-pro',
  })
  const after = JSON.parse(fs.readFileSync(configFile, 'utf-8'))
  assert(after.schemaVersion === 1, 'F: 最新版本号保持不变')
}

// 清理
try { fs.rmSync(tmp, { recursive: true, force: true }) } catch {}

if (failed > 0) process.exit(1)
console.log('\nAll config upgrade tests passed.')
