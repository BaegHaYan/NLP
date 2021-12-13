# Task
- AI 여친/카운슬러 -> '백하얀'
- 영상통화형 대화/공감/위로(멘탈웰빙)챗봇
- 사용자의 표정(감정)을 인식해 해당 대화에 맞는 발화를 함.
- 감정의 label은 [기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립]. 각 label을 특별토큰으로 만들어 문장의 앞에 넣어줌.

# data
- [챗봇 데이터](https://github.com/songys/Chatbot_data)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- [한국어 대화](https://aihub.or.kr/aidata/85/download)
- [한국어 대화 요약](https://aihub.or.kr/aidata/30714)
- [한국어 SNS](https://aihub.or.kr/aidata/30718)
- [멀티모달 영상](https://aihub.or.kr/aidata/137)
- 해당 데이터들을 전부 사용자 대화 앞에 감정 토큰을, 시스템 응답을 백하얀의 페르소나에 맞게 바꿔 사용함.
- 특별 토큰은 차례대로 `"[HAPPY]", "[PANIC]", "[ANGRY]", "[UNSTABLE]", "[HURT]", "[SAD]", "[NEUTRAL]"`.
- 각 데이터를 S1<s>S2</s>, R1</s> 형태로 바꿈(<s> - bos_token, </s> - eos_token).
- 각 데이터를 columns=["S1", "R1", "S2", "R3", "S3", "R3"]의 형태로 만들어 데이터셋 제작중에 합침 -> 알고리즘의 고안, 데이터의 row가 필요.
  

# 모델
- [byeongal/Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 를 파인튜닝해 사용 -> [cc-by-nc-sa-4.0] License(non-commercial)
- 토크나이저 파일을 모두 가져와 특별 토큰들을 추가해 토크나이저로 사용.
