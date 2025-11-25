# Patch to insert after line 262 in nodes.py

    # ------------------------------------------------------------------
    # 2.5 FORCE TOOL USAGE FOR INFORMATION QUESTIONS
    # ------------------------------------------------------------------
    
    # Check if model skipped tools for an information question
    has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
    
    if not has_tool_calls and user_message:
        # Detect information questions that MUST use retrieve_knowledge
        question_patterns = [
            'qual', 'quale', 'chi', 'cosa', 'dove', 'quando', 'perché', 'come',
            'mi sai dire', 'cos\'è', 'spiegami', 'che cosa', 'mail', 'email', 
            'telefono', 'phone', 'contatto'
        ]
        
        is_info_question = any(pattern in user_message.lower() for pattern in question_patterns)
        is_greeting = user_message.strip().lower() in ['ciao', 'hello', 'hi', 'buongiorno', 'come stai']
        
        if is_info_question and not is_greeting:
            logger.warning("⚠️ Model skipped tools for information question - forcing retrieve_knowledge")
            
            # Create a synthetic tool call
            import uuid
            from langchain_core.messages import AIMessage
            
            tool_call_id = str(uuid.uuid4())
            response = AIMessage(
                content="Sto cercando l'informazione richiesta.",
                tool_calls=[{
                    'name': 'retrieve_knowledge',
                    'args': {'query': user_message},
                    'id': tool_call_id
                }]
            )
            logger.info("✅ Forced retrieve_knowledge call")
