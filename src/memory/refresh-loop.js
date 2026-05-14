import { callLLM } from '../llm.js'
import { searchAdditionalMemories, formatMemoriesForPrompt } from './injector.js'

const WEB_KEYWORDS = /最新|实时|今天|昨天|明天|news|price|股价|天气|汇率|价格/i

function buildEvalPrompt(formattedMemories, query, { round = 1, prevMissing = [] } = {}) {
  const memSnippet = formattedMemories.slice(0, 600)
  const roundHint = round === 1
    ? `这是第1轮评估，基于当前已有的记忆片段作出判断。`
    : `这是第${round}轮评估。第${round - 1}轮识别的信息缺口是：${prevMissing.map(m => `"${m}"`).join('、') || '（无）'}。\n本轮追加注入了针对上述缺口专门检索的记忆片段，请定向利用这些新记忆重新评估。`
  return `你是一个记忆评估助手。${roundHint}根据提供的记忆片段，评估对以下问题的了解程度，输出 JSON。

已有记忆：
${memSnippet}

问题：${query}

只输出以下格式的 JSON，不要其他内容：
{"confidence":"low"|"medium"|"high","missing":["缺少的信息1","缺少的信息2"]}`
}

function parseEvalResult(content) {
  try {
    const match = content.match(/\{[\s\S]*?\}/)
    if (!match) throw new Error('no json')
    const parsed = JSON.parse(match[0])
    return {
      confidence: ['low', 'medium', 'high'].includes(parsed.confidence) ? parsed.confidence : 'medium',
      missing: Array.isArray(parsed.missing) ? parsed.missing : [],
    }
  } catch {
    return { confidence: 'medium', missing: [] }
  }
}

export async function runMemoryRefreshLoop({ originalQuery, baseMemories, systemPromptBase, formattedBaseMemories, signal }) {
  if (!originalQuery || !originalQuery.trim()) {
    return { additionalMemories: [], round3Results: '', roundsRun: 0, skipped: true }
  }

  let additionalMemories = []
  let round3Results = ''

  // 第1轮
  console.log('[记忆刷新] 第1轮 评估已有记忆覆盖度')
  let eval1 = { confidence: 'medium', missing: [] }
  try {
    if (signal?.aborted) return { additionalMemories, round3Results, roundsRun: 1, skipped: false }
    const sp1 = buildEvalPrompt(formattedBaseMemories, originalQuery, { round: 1 })
    const res1 = await callLLM({ systemPrompt: sp1, message: '请评估', maxTokens: 200, thinking: false, tools: [] })
    eval1 = parseEvalResult(res1.content || '')
  } catch (e) {
    console.log('[记忆刷新] 第1轮 LLM 调用失败:', e.message)
  }

  if (eval1.confidence === 'high') {
    return { additionalMemories, round3Results, roundsRun: 1, skipped: false }
  }

  // 第2轮：直接用第1轮识别的 missing 项作为搜索词（这才是"涌现的缺口概念"）
  console.log('[记忆刷新] 第2轮 针对缺口追加记忆召回')
  let eval2 = { confidence: 'medium', missing: eval1.missing }
  try {
    if (signal?.aborted) return { additionalMemories, round3Results, roundsRun: 2, skipped: false }
    const searchTerms = eval1.missing.slice(0, 6)
    if (searchTerms.length > 0) {
      const excludeIds = new Set(baseMemories.map(m => m.id))
      const newMemories = searchAdditionalMemories(searchTerms, excludeIds)
      if (newMemories.length > 0) {
        additionalMemories = newMemories
        const combinedFormatted = formattedBaseMemories + '\n\n' + formatMemoriesForPrompt([], newMemories)
        const sp2 = buildEvalPrompt(combinedFormatted, originalQuery, { round: 2, prevMissing: eval1.missing })
        const res2 = await callLLM({ systemPrompt: sp2, message: '请评估', maxTokens: 200, thinking: false, tools: [] })
        eval2 = parseEvalResult(res2.content || '')
      }
    }
  } catch (e) {
    console.log('[记忆刷新] 第2轮 LLM 调用失败:', e.message)
  }

  if (eval2.confidence === 'high') {
    return { additionalMemories, round3Results, roundsRun: 2, skipped: false }
  }

  // 第3轮
  console.log('[记忆刷新] 第3轮 针对 missing 发起外部查询')
  const missingItems = eval2.missing.slice(0, 3)
  const parts = []
  for (const item of missingItems) {
    if (signal?.aborted) break
    try {
      const needsWeb = WEB_KEYWORDS.test(item)
      const toolName = needsWeb ? 'web_search' : 'search_memory'
      const res3 = await callLLM({
        systemPrompt: systemPromptBase,
        message: `请搜索：${item}`,
        maxTokens: 600,
        thinking: false,
        tools: [toolName],
        signal,
      })
      const rawResult = (res3.toolResult?.result || res3.content || '').slice(0, 400)
      if (rawResult) parts.push(rawResult)
    } catch (e) {
      console.log(`[记忆刷新] 第3轮 "${item}" 查询失败:`, e.message)
    }
  }
  round3Results = parts.join('\n---\n')

  return { additionalMemories, round3Results, roundsRun: 3, skipped: false }
}
