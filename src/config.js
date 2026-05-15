import fs from 'fs'
import { paths } from './paths.js'

export const DEEPSEEK_PROVIDER = 'deepseek'
export const MINIMAX_PROVIDER = 'minimax'
export const OPENAI_PROVIDER = 'openai'
export const QWEN_PROVIDER = 'qwen'
export const MOONSHOT_PROVIDER = 'moonshot'
export const ZHIPU_PROVIDER = 'zhipu'

export const DEFAULT_DEEPSEEK_MODEL = 'deepseek-v4-flash'
export const DEFAULT_MINIMAX_MODEL = 'MiniMax-M2.7'
export const DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
export const DEFAULT_QWEN_MODEL = 'qwen-turbo'
export const DEFAULT_MOONSHOT_MODEL = 'moonshot-v1-8k'
export const DEFAULT_ZHIPU_MODEL = 'glm-4-flash'

export const DEEPSEEK_MODELS = [
  {
    id: 'deepseek-v4-flash',
    label: 'deepseek-v4-flash',
    deprecated: false,
  },
  {
    id: 'deepseek-v4-pro',
    label: 'deepseek-v4-pro',
    deprecated: false,
  },
  {
    id: 'deepseek-chat',
    label: 'deepseek-chat (将于 2026/07/24 弃用)',
    deprecated: true,
  },
  {
    id: 'deepseek-reasoner',
    label: 'deepseek-reasoner (将于 2026/07/24 弃用)',
    deprecated: true,
  },
]

export const MINIMAX_MODELS = [
  {
    id: 'MiniMax-M2.7',
    label: 'MiniMax-M2.7',
    deprecated: false,
  },
  {
    id: 'MiniMax-M1',
    label: 'MiniMax-M1',
    deprecated: false,
  },
]

export const OPENAI_MODELS = [
  {
    id: 'gpt-4o-mini',
    label: 'gpt-4o-mini',
    deprecated: false,
  },
  {
    id: 'gpt-4o',
    label: 'gpt-4o',
    deprecated: false,
  },
]

export const QWEN_MODELS = [
  {
    id: 'qwen-turbo',
    label: 'qwen-turbo',
    deprecated: false,
  },
  {
    id: 'qwen-plus',
    label: 'qwen-plus',
    deprecated: false,
  },
]

export const MOONSHOT_MODELS = [
  {
    id: 'moonshot-v1-8k',
    label: 'moonshot-v1-8k',
    deprecated: false,
  },
  {
    id: 'moonshot-v1-32k',
    label: 'moonshot-v1-32k',
    deprecated: false,
  },
]

export const ZHIPU_MODELS = [
  {
    id: 'glm-4-flash',
    label: 'glm-4-flash',
    deprecated: false,
  },
  {
    id: 'glm-4-plus',
    label: 'glm-4-plus',
    deprecated: false,
  },
]

const PROVIDER_CONFIG = {
  [DEEPSEEK_PROVIDER]: {
    label: 'DeepSeek',
    baseURL: 'https://api.deepseek.com',
    envVar: 'DEEPSEEK_API_KEY',
    models: DEEPSEEK_MODELS,
    defaultModel: DEFAULT_DEEPSEEK_MODEL,
  },
  [MINIMAX_PROVIDER]: {
    label: 'MiniMax',
    baseURL: 'https://api.minimax.chat/v1',
    envVar: 'MINIMAX_API_KEY',
    models: MINIMAX_MODELS,
    defaultModel: DEFAULT_MINIMAX_MODEL,
  },
  [OPENAI_PROVIDER]: {
    label: 'OpenAI',
    baseURL: 'https://api.openai.com/v1',
    envVar: 'OPENAI_API_KEY',
    models: OPENAI_MODELS,
    defaultModel: DEFAULT_OPENAI_MODEL,
  },
  [QWEN_PROVIDER]: {
    label: 'Qwen',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    envVar: 'DASHSCOPE_API_KEY',
    models: QWEN_MODELS,
    defaultModel: DEFAULT_QWEN_MODEL,
  },
  [MOONSHOT_PROVIDER]: {
    label: 'Moonshot',
    baseURL: 'https://api.moonshot.cn/v1',
    envVar: 'MOONSHOT_API_KEY',
    models: MOONSHOT_MODELS,
    defaultModel: DEFAULT_MOONSHOT_MODEL,
  },
  [ZHIPU_PROVIDER]: {
    label: 'Zhipu',
    baseURL: 'https://open.bigmodel.cn/api/paas/v4',
    envVar: 'ZHIPU_API_KEY',
    models: ZHIPU_MODELS,
    defaultModel: DEFAULT_ZHIPU_MODEL,
  },
}

const AUTO_PROVIDER = 'auto'
const PROBE_TIMEOUT_MS = 12000

function normalizeModel(model, provider = DEEPSEEK_PROVIDER) {
  const pConfig = PROVIDER_CONFIG[provider] || PROVIDER_CONFIG[DEEPSEEK_PROVIDER]
  const value = String(model || '').trim()
  const validIds = new Set(pConfig.models.map(m => m.id))
  if (validIds.has(value)) return value
  return pConfig.defaultModel
}

function isThinkingEnabledForModel(model) {
  return normalizeModel(model) !== 'deepseek-chat'
}

function getProvidersForAutoDetect() {
  return Object.entries(PROVIDER_CONFIG)
}

function getProviderErrorMessage(err) {
  const status = err?.status ?? err?.response?.status
  const message = err?.message || String(err)
  return status ? `${status} ${message}` : message
}

function withTimeout(promise, ms, label) {
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}

function buildPingParams(provider, model) {
  const pingParams = {
    model,
    messages: [{ role: 'user', content: 'Reply with exactly: hello' }],
    max_tokens: 8,
    temperature: 0,
    stream: false,
  }
  if (provider === DEEPSEEK_PROVIDER) {
    pingParams.reasoning_effort = 'high'
    pingParams.thinking = { type: isThinkingEnabledForModel(model) ? 'enabled' : 'disabled' }
  }
  return pingParams
}

async function probeProvider(OpenAI, provider, apiKey, requestedModel) {
  const pConfig = PROVIDER_CONFIG[provider]
  const model = normalizeModel(requestedModel, provider)
  const client = new OpenAI({
    apiKey,
    baseURL: pConfig.baseURL,
    timeout: PROBE_TIMEOUT_MS,
  })
  await withTimeout(
    client.chat.completions.create(buildPingParams(provider, model)),
    PROBE_TIMEOUT_MS,
    provider,
  )
  return { provider, model, pConfig }
}

async function detectProvider(OpenAI, apiKey, requestedModel) {
  const providers = getProvidersForAutoDetect()
  const errors = []

  return await new Promise((resolve, reject) => {
    let pending = providers.length
    for (const [provider] of providers) {
      probeProvider(OpenAI, provider, apiKey, requestedModel)
        .then(resolve)
        .catch((err) => {
          errors.push(`${provider}: ${getProviderErrorMessage(err)}`)
          pending -= 1
          if (pending === 0) {
            reject(new Error(`未识别出 API Key 所属厂商。已尝试: ${providers.map(([name]) => name).join(', ')}。最后错误: ${errors.slice(-3).join(' | ')}`))
          }
        })
    }
  })
}

function readStoredConfig() {
  try {
    if (!fs.existsSync(paths.configFile)) return null
    const raw = fs.readFileSync(paths.configFile, 'utf-8')
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return null
    if (!parsed.provider) return null
    if (parsed.provider === 'custom') {
      if (!parsed.baseURL || typeof parsed.baseURL !== 'string') return null
      if (!parsed.model || typeof parsed.model !== 'string') return null
      return parsed
    }
    if (!PROVIDER_CONFIG[parsed.provider]) return null
    if (!parsed.apiKey || typeof parsed.apiKey !== 'string') return null
    return parsed
  } catch {
    return null
  }
}

function writeStoredConfig(obj) {
  const tmp = paths.configFile + '.tmp'
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), 'utf-8')
  fs.renameSync(tmp, paths.configFile)
}

function shouldAllowEnvFallback() {
  return !process.versions?.electron
}

function loadFromEnv() {
  const deepseekKey = process.env['DEEPSEEK_API_KEY']
  if (deepseekKey) {
    return {
      provider: DEEPSEEK_PROVIDER,
      apiKey: deepseekKey,
      model: normalizeModel(process.env.DEEPSEEK_MODEL, DEEPSEEK_PROVIDER),
    }
  }
  const minimaxKey = process.env['MINIMAX_API_KEY']
  if (minimaxKey) {
    return {
      provider: MINIMAX_PROVIDER,
      apiKey: minimaxKey,
      model: normalizeModel(process.env.MINIMAX_MODEL, MINIMAX_PROVIDER),
    }
  }
  for (const [provider, pConfig] of Object.entries(PROVIDER_CONFIG)) {
    if (provider === DEEPSEEK_PROVIDER || provider === MINIMAX_PROVIDER) continue
    const key = process.env[pConfig.envVar]
    if (key) {
      return {
        provider,
        apiKey: key,
        model: normalizeModel(process.env[`${pConfig.envVar.replace(/_API_KEY$/, '')}_MODEL`], provider),
      }
    }
  }
  return null
}

function applyConfig(provider, apiKey, model, customBaseURL) {
  if (provider === 'custom') {
    config.provider = 'custom'
    config.model = String(model || '').trim()
    config.apiKey = apiKey || 'none'
    config.baseURL = String(customBaseURL || '').trim()
    config.needsActivation = false
    return
  }
  const pConfig = PROVIDER_CONFIG[provider]
  config.provider = provider
  config.model = normalizeModel(model, provider)
  config.apiKey = apiKey
  config.baseURL = pConfig.baseURL
  config.needsActivation = false
}

export const config = {
  tickInterval: 20 * 60 * 1000,
  provider: null,
  model: null,
  apiKey: null,
  baseURL: null,
  needsActivation: true,
  temperature: 0.5,
  security: {
    fileSandbox: true,
    execSandbox: true,
    blockedTools: [],
  },
}

const stored = readStoredConfig()
if (stored) {
  applyConfig(stored.provider, stored.apiKey, stored.model, stored.baseURL)
  if (typeof stored.temperature === 'number' && stored.temperature >= 0 && stored.temperature <= 2) {
    config.temperature = stored.temperature
  }
  if (stored.security && typeof stored.security === 'object') {
    if (typeof stored.security.fileSandbox === 'boolean') config.security.fileSandbox = stored.security.fileSandbox
    if (typeof stored.security.execSandbox === 'boolean') config.security.execSandbox = stored.security.execSandbox
    if (Array.isArray(stored.security.blockedTools)) config.security.blockedTools = stored.security.blockedTools
  }
} else if (shouldAllowEnvFallback()) {
  const fromEnv = loadFromEnv()
  if (fromEnv) applyConfig(fromEnv.provider, fromEnv.apiKey, fromEnv.model)
}

// 启动时把 config 文件里的 social 凭证写入 process.env，让各连接器能读到
;(function loadSocialEnv() {
  try {
    const raw = fs.readFileSync(paths.configFile, 'utf-8')
    const social = JSON.parse(raw)?.social || {}
    for (const [key, val] of Object.entries(social)) {
      if (typeof val === 'string' && val && globalThis.process?.env) {
        globalThis.process.env[key] = val
      }
    }
  } catch {}
})()

export async function activate({ provider = AUTO_PROVIDER, apiKey, model, baseURL }) {
  const p = String(provider || AUTO_PROVIDER).toLowerCase()

  if (p === 'custom') {
    const normalizedBaseURL = String(baseURL || '').trim()
    if (!normalizedBaseURL) throw new Error('自定义端点需要填写 Base URL')
    const normalizedModel = String(model || '').trim()
    if (!normalizedModel) throw new Error('自定义端点需要填写模型名称')
    const normalizedKey = String(apiKey || '').trim() || 'none'

    const { default: OpenAI } = await import('openai')
    const client = new OpenAI({ apiKey: normalizedKey, baseURL: normalizedBaseURL, timeout: PROBE_TIMEOUT_MS })
    try {
      await withTimeout(
        client.chat.completions.create({
          model: normalizedModel,
          messages: [{ role: 'user', content: 'Reply with exactly: hello' }],
          max_tokens: 16,
          temperature: 0,
          stream: false,
        }),
        PROBE_TIMEOUT_MS,
        'custom',
      )
    } catch (err) {
      const message = err?.message || String(err)
      throw new Error(`自定义端点连接失败: ${message}`)
    }

    applyConfig('custom', normalizedKey, normalizedModel, normalizedBaseURL)
    writeStoredConfig({
      provider: 'custom',
      apiKey: normalizedKey,
      model: normalizedModel,
      baseURL: normalizedBaseURL,
      activatedAt: new Date().toISOString(),
    })
    return {
      provider: 'custom',
      model: normalizedModel,
      models: [{ id: normalizedModel, label: normalizedModel, deprecated: false }],
    }
  }

  const pConfig = PROVIDER_CONFIG[p]
  if (p !== AUTO_PROVIDER && !pConfig) {
    throw new Error(`不支持的 provider: "${p}"，可选: ${Object.keys(PROVIDER_CONFIG).join(', ')}`)
  }

  const normalizedKey = String(apiKey || '').trim()
  const normalizedModel = normalizeModel(model, p)
  if (normalizedKey.length < 8) {
    throw new Error(`${p} Key 无效`)
  }

  const { default: OpenAI } = await import('openai')
  if (p === AUTO_PROVIDER) {
    const detected = await detectProvider(OpenAI, normalizedKey, model)
    applyConfig(detected.provider, normalizedKey, detected.model)
    writeStoredConfig({
      provider: detected.provider,
      apiKey: normalizedKey,
      model: detected.model,
      activatedAt: new Date().toISOString(),
    })
    return {
      provider: detected.provider,
      model: detected.model,
      models: detected.pConfig.models,
    }
  }

  const client = new OpenAI({ apiKey: normalizedKey, baseURL: pConfig.baseURL, timeout: PROBE_TIMEOUT_MS })

  try {
    await withTimeout(
      client.chat.completions.create(buildPingParams(p, normalizedModel)),
      PROBE_TIMEOUT_MS,
      p,
    )
  } catch (err) {
    const message = err?.message || String(err)
    if (/401|unauthoriz|invalid.*api.*key|authentication/i.test(message)) {
      throw new Error(`${p} Key 校验失败，请确认 key 是否正确`)
    }
    throw new Error(`${p} 验证失败: ${message}`)
  }

  applyConfig(p, normalizedKey, normalizedModel)
  writeStoredConfig({
    provider: p,
    apiKey: normalizedKey,
    model: normalizedModel,
    activatedAt: new Date().toISOString(),
  })

  return {
    provider: p,
    model: normalizedModel,
    models: pConfig.models,
  }
}

export function getActivationStatus() {
  const pConfig = config.provider && config.provider !== 'custom' ? PROVIDER_CONFIG[config.provider] : null
  const customModels = config.model ? [{ id: config.model, label: config.model, deprecated: false }] : DEEPSEEK_MODELS
  return {
    activated: !config.needsActivation,
    provider: config.provider,
    model: config.model,
    baseURL: config.provider === 'custom' ? config.baseURL : undefined,
    models: pConfig ? pConfig.models : customModels,
    defaultModel: pConfig ? pConfig.defaultModel : (config.model || DEFAULT_DEEPSEEK_MODEL),
  }
}

export function getProviderSummaries() {
  const result = Object.fromEntries(Object.entries(PROVIDER_CONFIG).map(([name, pConfig]) => [
    name,
    {
      label: pConfig.label || name,
      models: pConfig.models,
      defaultModel: pConfig.defaultModel,
    },
  ]))
  result.custom = { label: '自定义端点', models: [], defaultModel: '' }
  return result
}

export function deactivate() {
  try {
    if (fs.existsSync(paths.configFile)) fs.unlinkSync(paths.configFile)
  } catch {}
  config.provider = null
  config.model = null
  config.apiKey = null
  config.baseURL = null
  config.needsActivation = true
}

export function switchModel(model) {
  if (!config.apiKey) throw new Error('尚未激活，无法切换模型')
  if (config.provider === 'custom') {
    const trimmed = String(model || '').trim()
    if (!trimmed) throw new Error('模型名称不能为空')
    config.model = trimmed
    try {
      const existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))
      writeStoredConfig({ ...existing, model: trimmed })
    } catch {}
    return { provider: 'custom', model: trimmed }
  }
  const normalized = normalizeModel(model, config.provider)
  config.model = normalized
  try {
    const existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))
    writeStoredConfig({ ...existing, model: normalized })
  } catch {}
  return { provider: config.provider, model: normalized }
}

export function setTemperature(t) {
  const v = Math.min(2, Math.max(0, Number(t) || 0.5))
  config.temperature = v
  try {
    const existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))
    writeStoredConfig({ ...existing, temperature: v })
  } catch {}
  return { temperature: v }
}

export function getSecurity() {
  return {
    fileSandbox: config.security.fileSandbox,
    execSandbox: config.security.execSandbox,
    blockedTools: [...config.security.blockedTools],
  }
}

export function setSecurity(updates) {
  if (typeof updates.fileSandbox === 'boolean') config.security.fileSandbox = updates.fileSandbox
  if (typeof updates.execSandbox === 'boolean') config.security.execSandbox = updates.execSandbox
  if (Array.isArray(updates.blockedTools)) {
    config.security.blockedTools = updates.blockedTools.filter(t => typeof t === 'string')
  }
  try {
    const existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))
    writeStoredConfig({ ...existing, security: { ...config.security } })
  } catch {}
  return getSecurity()
}

export function getMinimaxKey() {
  try {
    const raw = fs.readFileSync(paths.configFile, 'utf-8')
    const parsed = JSON.parse(raw)
    return typeof parsed?.minimax_api_key === 'string' ? parsed.minimax_api_key : null
  } catch { return null }
}

export function setMinimaxKey(key) {
  const trimmed = String(key || '').trim()
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  if (trimmed) {
    writeStoredConfig({ ...existing, minimax_api_key: trimmed })
  } else {
    const { minimax_api_key: _removed, ...rest } = existing
    writeStoredConfig(rest)
  }
}

// ── Social media platform config ──

const SOCIAL_ENV_KEYS = [
  'DISCORD_BOT_TOKEN',
  'FEISHU_APP_ID', 'FEISHU_APP_SECRET', 'FEISHU_VERIFICATION_TOKEN',
  'WECHAT_OFFICIAL_APP_ID', 'WECHAT_OFFICIAL_APP_SECRET', 'WECHAT_OFFICIAL_TOKEN',
  'WECOM_BOT_KEY', 'WECOM_INCOMING_TOKEN',
]

// ── WeChat ClawBot 凭证（扫码后自动写入，不暴露到 SOCIAL_ENV_KEYS）──

export function getClawbotCredentials() {
  try {
    const stored = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))
    const c = stored?.clawbot
    return (c?.accountId && c?.botToken) ? c : null
  } catch { return null }
}

export function setClawbotCredentials({ accountId, botToken, baseUrl }) {
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  writeStoredConfig({ ...existing, clawbot: { accountId, botToken, baseUrl } })
}

export function clearClawbotCredentials() {
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  const { clawbot: _, ...rest } = existing
  writeStoredConfig(rest)
}

export function getSocialConfig() {
  let stored = {}
  try { stored = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))?.social || {} } catch {}
  const result = {}
  for (const key of SOCIAL_ENV_KEYS) {
    const val = stored[key] || globalThis.process?.env?.[key] || ''
    result[key] = { configured: !!val }
  }
  return result
}

export function setSocialConfig(updates) {
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  const current = existing.social || {}
  const next = { ...current }
  for (const [key, val] of Object.entries(updates)) {
    if (!SOCIAL_ENV_KEYS.includes(key)) continue
    const trimmed = String(val || '').trim()
    if (trimmed) {
      next[key] = trimmed
      // 立即生效，无需重启
      if (globalThis.process?.env) globalThis.process.env[key] = trimmed
    } else {
      delete next[key]
    }
  }
  writeStoredConfig({ ...existing, social: next })
}

const VOICE_CONFIG_KEYS = ['aliyunApiKey', 'tencentSecretId', 'tencentSecretKey', 'tencentAppId', 'xunfeiAppId', 'xunfeiApiKey', 'xunfeiApiSecret']

export function getVoiceConfig() {
  let stored = {}
  try { stored = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))?.voice || {} } catch {}
  const result = {}
  for (const key of VOICE_CONFIG_KEYS) {
    result[key] = { configured: !!(stored[key]) }
  }
  return result
}

export function setVoiceConfig(updates) {
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  const current = existing.voice || {}
  const next = { ...current }
  for (const [key, val] of Object.entries(updates)) {
    if (!VOICE_CONFIG_KEYS.includes(key)) continue
    const trimmed = String(val || '').trim()
    if (trimmed) next[key] = trimmed
    else delete next[key]
  }
  writeStoredConfig({ ...existing, voice: next })
}

// TTS 配置
const TTS_CONFIG_KEYS = [
  'ttsProvider', 'ttsVoiceId',
  'minimaxKey',
  'doubaoKey', 'doubaoAppId', 'doubaoAccessKey', 'doubaoResourceId',
  'openaiTtsKey', 'openaiTtsBaseURL',
  'elevenLabsKey',
  'volcanoAppId', 'volcanoToken',
]

export function getTTSConfig() {
  let stored = {}
  try { stored = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))?.tts || {} } catch {}
  return {
    ttsProvider:     stored.ttsProvider  || 'doubao',
    ttsVoiceId:      stored.ttsVoiceId   || 'zh_female_xiaohe_uranus_bigtts',
    minimaxKey:      { configured: !!(stored.minimaxKey || process.env.MINIMAX_API_KEY || getMinimaxKey()) },
    doubaoKey:       { configured: !!(stored.doubaoKey) },
    doubaoAppId:     { configured: !!(stored.doubaoAppId), value: stored.doubaoAppId || '' },
    doubaoAccessKey: { configured: !!(stored.doubaoAccessKey) },
    doubaoResourceId: stored.doubaoResourceId || '',
    openaiTtsBaseURL: stored.openaiTtsBaseURL || '',
    openaiTtsKey:    { configured: !!(stored.openaiTtsKey) },
    elevenLabsKey:   { configured: !!(stored.elevenLabsKey) },
    volcanoAppId:    { configured: !!(stored.volcanoAppId), value: stored.volcanoAppId || '' },
    volcanoToken:    { configured: !!(stored.volcanoToken) },
  }
}

// 读取明文 TTS 凭证（仅供后端调用 TTS API 使用，不暴露给前端）
export function getTTSCredentials() {
  let stored = {}
  try { stored = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8'))?.tts || {} } catch {}
  return {
    provider:       stored.ttsProvider  || 'doubao',
    voiceId:        stored.ttsVoiceId   || 'zh_female_xiaohe_uranus_bigtts',
    doubaoKey:      stored.doubaoKey    || process.env.DOUBAO_TTS_API_KEY || '',
    doubaoAppId:    stored.doubaoAppId  || process.env.DOUBAO_TTS_APP_ID || '',
    doubaoAccessKey: stored.doubaoAccessKey || process.env.DOUBAO_TTS_ACCESS_KEY || '',
    doubaoResourceId: stored.doubaoResourceId || process.env.DOUBAO_TTS_RESOURCE_ID || '',
    minimaxKey:     process.env.MINIMAX_API_KEY || stored.minimaxKey || getMinimaxKey() || (config.provider === 'minimax' ? config.apiKey : '') || '',
    openaiKey:      stored.openaiTtsKey  || '',
    openaiBaseURL:  stored.openaiTtsBaseURL || '',
    elevenLabsKey:  stored.elevenLabsKey || '',
    volcanoAppId:   stored.volcanoAppId  || '',
    volcanoToken:   stored.volcanoToken  || '',
  }
}

export function setTTSConfig(updates) {
  let existing = {}
  try { existing = JSON.parse(fs.readFileSync(paths.configFile, 'utf-8')) } catch {}
  const current = existing.tts || {}
  const next = { ...current }
  for (const [key, val] of Object.entries(updates)) {
    if (!TTS_CONFIG_KEYS.includes(key)) continue
    const trimmed = String(val || '').trim()
    if (trimmed) next[key] = trimmed
    else delete next[key]
  }
  writeStoredConfig({ ...existing, tts: next })
}

export const __internals = {
  DEEPSEEK_MODELS,
  MINIMAX_MODELS,
  OPENAI_MODELS,
  QWEN_MODELS,
  MOONSHOT_MODELS,
  ZHIPU_MODELS,
  normalizeModel,
  isThinkingEnabledForModel,
  buildPingParams,
}
