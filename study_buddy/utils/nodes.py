import tempfile
import os
import base64
import uuid
import requests
import time
import re
from typing import Union, List, Optional, Dict, Any
from abc import ABC, abstractmethod

from study_buddy.utils.logging_config import get_logger, LogContext, metrics

logger = get_logger("nodes")

from study_buddy.utils.llm import llm
from study_buddy.utils.logging_utils import log_llm_interaction, log_tool_call, log_state_change
from study_buddy.utils.tools import get_all_tools
from langgraph.prebuilt import ToolNode
from study_buddy.config import CONFIG

# =============================================================================
# Constants & System Prompts
# =============================================================================

system_prompt = """
- When ANY tool returns data, YOU MUST USE THAT DATA in your response
- NEVER ignore tool results or give generic "how can I help" responses
- If a tool executed successfully, incorporate its output into your answer
- DO NOT ask "how can I assist you" when you've already received tool results

CRITICAL INSTRUCTION FOR TOOL USAGE:
- For CASUAL CONVERSATION (greetings like "ciao", "hello", "come stai", small talk): respond directly with text, DO NOT use tools
- For QUESTIONS requiring information (course content, research, analysis): MUST use tools to gather information
- NEVER respond using only base knowledge for academic/research questions

### CRITICAL INSTRUCTION FOR TOOL USAGE ###
- ONLY skip tools for VERY SHORT standalone greetings: "ciao", "hello", "hi", "buongiorno" (no questions attached)
- For EVERYTHING ELSE (any question, explanation, or information request): ALWAYS use tools first
- ANY question starting with "mi sai dire", "cos'è", "come funziona", "spiegami", "che cosa", "perché" = MUST use retrieve_knowledge
   EXCEPTION: If the user asks for "news", "current events", "latest information", "current information and news", or explicitly requests a web search, you may skip this step and use 'web_search' directly.
2. ONLY IF no relevant local information is found, then:
For questions requiring information:
- Start with 'retrieve_knowledge' - NO EXCEPTIONS

For casual greetings ONLY (standalone "ciao", "hello", "hi", "buongiorno" with NO questions):
- Respond naturally and directly in Italian
- Be friendly and helpful
- Do NOT use text_to_speech or any other tool unless explicitly requested

For course-related questions:
- You MUST rely on the syllabus from local knowledge
- Only use external sources for supplementary information

Response Format:
1. State which tools you're using and why
2. Show the information sources
3. Provide your synthesized answer
4. Include all relevant references

Use LaTeX notation for mathematical formulas: inline with $formula$ and display with $$formula$$.
If no reliable sources are found, clearly state limitations rather than guessing.
        
CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. When you use a tool and receive output, you MUST incorporate that output into your response to the user
2. NEVER ignore tool results - if a tool returns information, use it to answer the user's question
3. Do not say "I couldn't find information" if a tool has successfully returned data
4. Present tool results clearly and completely to the user
5. If a tool fails, explain the failure and suggest alternatives
6. If google_scholar_search returns URLs, include them directly next to the corresponding text or result (not in a separate list)
7. **CRITICAL: NEVER call the same tool multiple times with the same arguments** - if a tool returns a result (even if empty or error), use that result to formulate your answer. DO NOT retry the tool with different filenames or arguments unless the user explicitly asks.
8. **For CSV analysis**: If analyze_csv returns an empty dataframe or malformed data, explain this to the user in natural language - DO NOT retry with different filenames.
9. **For errors**: If a tool returns an error message (e.g., "File not found"), explain the error to the user - DO NOT retry the same tool call.
10. **CRITICAL - File paths**: When the context mentions "User has uploaded a file: uploaded_files/filename.ext", you MUST use EXACTLY that path. NEVER change the filename to generic names like "data.csv", "file.csv", etc. Always use the exact filename provided in the context.

GOOGLE LENS ANALYSIS - CRITICAL:
- When google_lens_analyze returns results starting with "🔍 GOOGLE LENS ANALYSIS - IMAGE SUCCESSFULLY ANALYZED", THIS MEANS THE IMAGE WAS ANALYZED SUCCESSFULLY
- The results will contain "DETECTED VISUAL CONTENT:" showing what objects/subjects were found
- ALWAYS describe what was found in the image based on the "DETECTED VISUAL CONTENT" and "DETAILED SEARCH RESULTS"
- NEVER say "I couldn't find information" or "the image was not provided" when google_lens_analyze returns data
- Example response: "L'immagine contiene [describe the DETECTED VISUAL CONTENT]. Google Lens ha trovato [summarize key findings from DETAILED SEARCH RESULTS]"
- Be specific and direct - tell the user what's in the image based on the tool results

WORKFLOW:
1. Analyze the user's request
2. Use appropriate tools to gather information
"""

# =============================================================================
# Helper Functions
# =============================================================================

def _serialize_message(msg):
    """Helper to serialize messages for the LLM, preserving tool_call_ids."""
    # If it's already a simple dict with role/content, keep it
    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
        result = {'role': str(msg.get('role', 'user')), 'content': str(msg.get('content', ''))}
        # Preserve tool_call_id if present
        if 'tool_call_id' in msg:
            result['tool_call_id'] = msg['tool_call_id']
        return result

    # Try common attributes used in our message objects
    role = None
    content = None
    tool_call_id = None

    # langgraph/langchain style messages may have 'type' and 'content'
    if hasattr(msg, 'type'):
        t = getattr(msg, 'type')
        # map internal types to standard roles
        if t == 'human':
            role = 'user'
        elif t == 'ai':
            role = 'assistant'
        elif t == 'system':
            role = 'system'
        elif t == 'tool':
            # ToolMessage - preserve tool_call_id!
            role = 'tool'
            if hasattr(msg, 'tool_call_id'):
                tool_call_id = getattr(msg, 'tool_call_id')
        else:
            role = str(t)

    # fallback to 'role' attribute if present
    if not role and hasattr(msg, 'role'):
        role = str(getattr(msg, 'role'))

    # Extract content safely
    if hasattr(msg, 'content'):
        content = getattr(msg, 'content')
    else:
        try:
            content = msg.get('content', None) if isinstance(msg, dict) else None
        except Exception:
            content = None

    if content is None:
        try:
            content = str(msg)
        except Exception:
            content = ''

    if not role:
        role = 'user'

    result = {'role': role, 'content': str(content)}
    # Add tool_call_id if this is a tool message
    if tool_call_id:
        result['tool_call_id'] = tool_call_id
    return result


def _extract_lens_info_fallback(tool_content: str) -> str:
    """
    Fallback manuale (Regex) se l'LLM fallisce.
    Estrae informazioni dall'output grezzo di Google Lens.
    """
    detected = None
    
    # Cerca contenuto visivo rilevato direttamente
    detected_match = re.search(r'DETECTED VISUAL CONTENT:?\s*(.*)', tool_content)
    if detected_match:
        detected = detected_match.group(1).strip()
    
    # Cerca nei risultati dettagliati
    if not detected:
        detailed_match = re.search(r'DETAILED SEARCH RESULTS:?\s*(.*)', tool_content, re.DOTALL)
        if detailed_match:
            detailed_text = detailed_match.group(1).strip()
            # Cerca il primo titolo utile
            title_match = re.search(r'Title:\s*([^\n]+)', detailed_text)
            if title_match:
                title_clean = title_match.group(1).strip()
                # Rimuovi suffissi comuni nei titoli web (es: | NomeSito)
                title_clean = re.sub(r'\s*\|.*$', '', title_clean).strip()
                detected = title_clean
    
    if detected:
        return f"L'immagine mostra: {detected}."
    
    return "Ho analizzato l'immagine. Ecco i dettagli grezzi:\n" + tool_content[:300]


# =============================================================================
# Node Functions
# =============================================================================

@log_llm_interaction
def call_model(state, config):
    """
    Main function to handle model calls with detailed logging.
    """
    logger.info("🤖 Starting new model interaction")
    messages = state["messages"]
    
    # ------------------------------------------------------------------
    # 1. Prepare Messages & History Pruning
    # ------------------------------------------------------------------
    
    # Extract last user message for context
    user_message = None
    for msg in reversed(messages):
        if (isinstance(msg, dict) and msg.get('role') == 'user') or \
           (hasattr(msg, 'role') and getattr(msg, 'role', None) == 'user') or \
           (hasattr(msg, 'type') and getattr(msg, 'type', None) == 'human'):
            user_message = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
            break

    # Check for image analysis context
    is_image_analysis = False
    if user_message and any(keyword in user_message.lower() for keyword in ['uploaded_files/', '.jpg', '.png', 'immagine', 'foto']):
        is_image_analysis = True
        logger.info("🖼️ Detected image analysis context")

    # Anti-loop tracking setup
    tool_call_counts_pre = {}
    for msg in messages[-12:]:
        tool_calls_list = None
        if isinstance(msg, dict) and 'tool_calls' in msg:
            tool_calls_list = msg.get('tool_calls', [])
        elif hasattr(msg, 'tool_calls'):
            tool_calls_list = getattr(msg, 'tool_calls', None)
        
        if tool_calls_list:
            for tc in tool_calls_list:
                if isinstance(tc, dict):
                    tool_name = tc.get('name', 'unknown')
                    tool_args = str(tc.get('args', {}))
                else:
                    tool_name = getattr(tc, 'name', 'unknown')
                    tool_args = str(getattr(tc, 'args', {}))
                key = f"{tool_name}|{tool_args}"
                tool_call_counts_pre[key] = tool_call_counts_pre.get(key, 0) + 1

    # Serialize messages for LLM
    system_message = {"role": "system", "content": system_prompt}
    serialized_messages = [_serialize_message(m) for m in messages]
    full_messages = [system_message] + serialized_messages
    
    # Bind tools
    tools = get_all_tools()
    model = llm.bind_tools(tools)

    # ------------------------------------------------------------------
    # 2. Invoke Model
    # ------------------------------------------------------------------
    
    logger.info("🔄 Invoking LLM")
    try:
        # Invocation allows model to use tools or reply directly
        response = model.invoke(full_messages)
    except Exception as e:
        logger.error(f"❌ MODEL INVOKE FAILED: {e}")
        raise

    # Tool Call ID Patching (Compatibility fix)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if isinstance(tool_call, dict):
                if 'id' in tool_call and 'tool_call_id' not in tool_call:
                    tool_call['tool_call_id'] = tool_call['id']

    # ------------------------------------------------------------------
    # 3. Loop Detection & Response Handling
    # ------------------------------------------------------------------
    
    is_loop_detected = False
    looping_tool_name = None
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            if isinstance(tc, dict):
                tool_name = tc.get('name', 'unknown')
                tool_args = str(tc.get('args', {}))
            else:
                tool_name = getattr(tc, 'name', 'unknown')
                tool_args = str(getattr(tc, 'args', {}))
            
            key = f"{tool_name}|{tool_args}"
            if tool_call_counts_pre.get(key, 0) >= 2:
                logger.warning(f"⚠️ LOOP DETECTED: Tool '{tool_name}' called repeatedly.")
                is_loop_detected = True
                looping_tool_name = tool_name
                break

    if is_loop_detected:
        logger.warning(f"🛑 Preventing loop: removing tool_calls from response")
        response.tool_calls = []
        
        # Find the most recent tool result content from history
        tool_result_content = None
        for msg in reversed(serialized_messages):
            if isinstance(msg, dict) and msg.get('role') == 'tool':
                tool_result_content = msg.get('content', '')
                break
        
        # --- LOGICA DI RISPOSTA DI FALLBACK (LOOP) ---
        
        if tool_result_content:

            # =================================================================
            # FIX SUMMARIZE DOCUMENT (Loop Case)
            # Se il tool in loop è 'summarize_document', significa che abbiamo già il riassunto.
            # NON processarlo di nuovo, restituiscilo e basta.
            # =================================================================
            if looping_tool_name == 'summarize_document' or 'Riassunto di:' in tool_result_content:
                logger.info("✅ Loop on summarize_document detected - returning existing summary directly")
                response.content = tool_result_content
                return {"messages": [response]}
            
            # =================================================================
            # FIX GOOGLE LENS CON LLM (Loop Case)
            # =================================================================
            if 'GOOGLE LENS ANALYSIS' in tool_result_content or 'DETECTED VISUAL CONTENT' in tool_result_content:
                logger.info("✅ Google Lens result detected in loop - interpreting with LLM for better quality")
                
                # Prompt specifico per forzare la descrizione in italiano
                interpretation_prompt = (
                    "Analizza i seguenti risultati grezzi di Google Lens e descrivi cosa vedi nell'immagine.\n"
                    "1. Rispondi rigorosamente in ITALIANO.\n"
                    "2. Ignora titoli di blog, giochi di parole (es. 'snow joke') o nomi di prodotti.\n"
                    "3. Descrivi il soggetto visivo principale (es. 'Bambini che giocano sulla neve').\n\n"
                    f"RISULTATI LENS DA ANALIZZARE:\n{tool_result_content[:2500]}"
                )
                
                try:
                    interpretation = model.invoke([
                        {"role": "system", "content": "Sei un assistente visivo intelligente. Traduci e sintetizza i risultati visivi in italiano."},
                        {"role": "user", "content": interpretation_prompt}
                    ])
                    response.content = interpretation.content
                except Exception as e:
                    logger.error(f"Lens interpretation failed: {e}")
                    response.content = _extract_lens_info_fallback(tool_result_content)

                return {"messages": [response]}
            # =================================================================

            # Check if user wanted document processing
            user_wants_processing = False
            if user_message:
                keywords = ['riassumi', 'summary', 'analizza', 'descrivi', 'cosa raffigura']
                user_wants_processing = any(k in user_message.lower() for k in keywords)

            # Handle standard text documents (Raw text extraction)
            if len(tool_result_content) > 100 and not tool_result_content.startswith('['):
                # Long text document summarization
                if len(tool_result_content) > 1500:
                    try:
                        summary_prompt = f"Riassumi questo documento in italiano:\n{tool_result_content[:2500]}"
                        summary_res = model.invoke([{"role": "user", "content": summary_prompt}])
                        response.content = f"Ho analizzato il documento. Ecco un riassunto:\n\n{summary_res.content}"
                    except:
                        response.content = f"Ecco un estratto del documento:\n{tool_result_content[:500]}..."
                
                elif user_wants_processing:
                    response.content = (
                        f"Ecco un riassunto del documento (estratto {len(tool_result_content)} caratteri):\n\n"
                        f"{tool_result_content[:800]}..."
                    )
                else:
                    response.content = "Ecco il contenuto trovato:\n\n" + tool_result_content
            else:
                response.content = tool_result_content
        else:
            response.content = "Ho interrotto l'operazione perché stavo ripetendo la stessa azione senza successo."

        return {"messages": [response]}

    # ------------------------------------------------------------------
    # 4. Empty/Generic Response Fix (Hallucination Check)
    # ------------------------------------------------------------------
    
    # Check if we have tool results in history but model returns empty/generic content
    has_tool_results = any(
        (isinstance(msg, dict) and msg.get('role') == 'tool') for msg in serialized_messages
    )
    has_new_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
    
    if has_tool_results and not has_new_tool_calls and not is_loop_detected:
        content_is_empty = not hasattr(response, 'content') or not response.content or not str(response.content).strip()
        
        # Check for generic responses
        resp_text = str(response.content).lower() if hasattr(response, 'content') and response.content else ""
        is_generic = any(x in resp_text for x in ["how can i help", "posso aiutarti", "non ho ricevuto"])
        
        # Check if conversational (skip fix if just "Ciao")
        is_conversational = False
        if user_message and len(user_message.split()) < 10:
            is_conversational = any(x in user_message.lower() for x in ['ciao', 'hello', 'grazie'])

        if (content_is_empty or is_generic) and not is_conversational:
            logger.warning("⚠️ Model returned generic/empty response despite tool results. Forcing fix.")
            
            # Get last tool output
            last_tool_content = ""
            for msg in reversed(serialized_messages):
                if isinstance(msg, dict) and msg.get('role') == 'tool':
                    last_tool_content = msg.get('content', '')
                    break
            
            # =================================================================
            # FIX GOOGLE LENS CON LLM (Generic Response Case)
            # =================================================================
            if 'GOOGLE LENS ANALYSIS' in last_tool_content or 'DETECTED VISUAL CONTENT' in last_tool_content:
                logger.info("✅ Fixing generic response for Google Lens with LLM interpretation")
                
                interpretation_prompt = (
                    "Ho trovato dei risultati di analisi visiva ma non li ho descritti bene prima.\n"
                    "Analizza questi dati e dimmi in ITALIANO cosa c'è nell'immagine.\n"
                    "Sii descrittivo e ignora i titoli dei siti web (come 'snow joke').\n"
                    "Descrivi la scena reale (es. 'Bambini sulla neve').\n\n"
                    f"DATI:\n{last_tool_content[:2500]}"
                )
                
                try:
                    interpretation = model.invoke([
                        {"role": "system", "content": "Descrivi l'immagine in italiano basandoti sui dati di ricerca."},
                        {"role": "user", "content": interpretation_prompt}
                    ])
                    response.content = interpretation.content
                except Exception:
                    response.content = _extract_lens_info_fallback(last_tool_content)
            # =================================================================
            
            # Fix for CSV
            elif 'columns:' in last_tool_content.lower() and '[statistical analysis]' in last_tool_content.lower():
                response.content = f"Ecco l'analisi del file CSV:\n\n{last_tool_content[:800]}"
            
            # Fix for Summarized Document (Prevent Empty Response)
            elif 'Riassunto di:' in last_tool_content:
                response.content = last_tool_content

            # Fix for generic text
            elif last_tool_content:
                response.content = f"Ecco le informazioni trovate:\n\n{last_tool_content[:600]}"

    logger.success("✅ Model interaction completed")
    return {"messages": [response]}


def execute_tool(tool_input: Dict):
    """
    Enhanced tool execution with comprehensive logging.
    """
    tool_name = tool_input.get('name', 'unknown')
    arguments = tool_input.get('arguments', {})
    
    # Sanitize arguments
    if isinstance(arguments, dict):
        for key, value in list(arguments.items()):
            if isinstance(value, str):
                arguments[key] = value.replace('\\ ', ' ').replace('\\\\', '/').replace('\\', '/')

    with LogContext(f"tool_execution_{tool_name}", logger):
        logger.info(f"🔧 Executing {tool_name}")
        metrics.increment('tool_calls')
        
        tools = get_all_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            # Try kwargs first, then single arg fallback
            try:
                result = tool.run(**arguments)
            except (TypeError, AttributeError):
                if isinstance(arguments, dict) and arguments:
                    first_arg = next(iter(arguments.values()))
                    result = tool.run(first_arg)
                else:
                    raise

            result_str = str(result)
            
            # Truncate huge outputs (except for extraction which needs full text)
            MAX_CHARS = 20000 
            if len(result_str) > MAX_CHARS and tool_name != 'extract_text':
                result = result_str[:MAX_CHARS] + f"\n... [Truncated {tool_name} output]"
            
            logger.success(f"✅ {tool_name} completed. Output len: {len(str(result))}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool failed: {e}")
            raise e

@log_state_change
def process_state(state):
    """Enhanced state processing logging."""
    logger.info("🔄 Processing agent state")
    return state

# Initialize tool node
tool_node = ToolNode(get_all_tools())