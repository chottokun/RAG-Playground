def reflection_component(state, **kwargs):
    """
    十分な答えが得られていれば'sufficient'、そうでなければ'insufficient'を返す。
    """
    answer = state.get('final_answer') or state.get('answer')
    # 10文字以上の答えがあれば十分とみなす例
    if answer and len(str(answer).strip()) > 10:
        state['reflection_result'] = 'sufficient'
    else:
        state['reflection_result'] = 'insufficient'
    return {'reflection_result': state['reflection_result']}
