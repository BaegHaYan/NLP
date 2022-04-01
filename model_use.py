from typing import List, Dict, Union
from . import *

def chat_with_bot(
        user_text: str,
        user_emotion_id: int,
        past_convs: List[str],
        DB_QnA: Dict[str, str]) -> str:
    """
    chat with bot

    Examples
         >>> example_text = "잘 지냈어?"
         >>> example_emotion_id = 0  # [HAPPY]
         >>> example_past_convs = ["['NATURAL']안녕?", "안녕."]
         >>> example_QnA = {"넌 몇살이야?": "난 올해 스무살이야."}

         >>> bot_response = chat_with_bot(example_text, example_emotion_id, example_past_convs, example_QnA)
         >>> print(bot_response)  # 응. 잘 지냈어. (can be different)


    Args:
        user_text : the user spoken text.
        user_emotion_id : the Id for emotion of user. following the label_dict.
        past_convs : the List including past convs. it has to order by time.
        DB_QnA : the Dict including prepared Questions(Key) and Answers(Value).
    """

    # searching question in prepared questions
    searched_answer = searching_QnA(user_text, DB_QnA)
    if searched_answer is not None:
        return searched_answer
    # check the user_input has hate expression
    pass
    # tokenizing user_text and past conversations(last 5 turn(10 sent))
    emotion_token = label_dict[user_emotion_id]
    user_text = tokenizer_for_chat.encode(emotion_token + user_text, return_tensors="pt").to(device)
    for i in range(len(past_convs)-1, len(past_convs)-(USING_TURN*2)-1, -1):
        if i < 0:
            break
        encoded_vector = tokenizer_for_chat.encoding(past_convs[i] + tokenizer_for_chat.eos_token)
        user_text = torch.cat([encoded_vector, user_text], dim=-1)
    # generate bot answer
    output = chat_model.generate(user_text,
                                 max_length=1000,
                                 num_beams=5,
                                 top_k=20,
                                 no_repeat_ngram_size=4,
                                 length_penalty=0.65,
                                 repetition_penalty=2.)
    bot_output = tokenizer_for_chat.decode(output[0][user_text.shape[-1]:])
    # # check the generated answer has correct persona(returned persona percentage of sentence)
    # if persona_classifier.predict(bot_output) < 0.2:
    #     # if bot output does not have enough persona, change output using model
    #     bot_text = tokenizer_for_persona.encode(bot_output, return_tensors="pt").to(device)
    #     output = persona_converter.generate(bot_text,
    #                                         max_length=1000,
    #                                         num_beams=5,
    #                                         top_k=20,
    #                                         no_repeat_ngram_size=4,
    #                                         length_penalty=0.65,
    #                                         repetition_penalty=2.)
    #     bot_output = tokenizer_for_persona.decode(output[0][bot_text.shape[-1]:])
    return bot_output

def searching_QnA(user_text: str, DB_QnA: Dict[str, str]) -> Union[str, None]:
    min_sim = 0.8
    searched_answer = None
    user_text = embedding_sent(user_text)

    for question, answer in DB_QnA.items():
        temp_sim = cos_sim(user_text, embedding_sent(question))
        if temp_sim > min_sim:
            min_sim = temp_sim
            searched_answer = answer
    return searched_answer

def embedding_sent(sent: str) -> List[float]:
    sent = tokenizer_for_chat.encode(sent, return_tensors="pt")
    return embedding_model(sent)

def cos_sim(v1: List[float], v2: List[float]) -> np.float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
