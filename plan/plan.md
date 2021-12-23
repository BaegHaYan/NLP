# Task
- AI 여친/카운슬러 -> '백하얀'
- 영상통화형 대화/공감/위로(멘탈웰빙)챗봇
- 사용자의 표정(감정)을 인식해 해당 대화에 맞는 발화를 함.
- 감정의 label은 [기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립]. 각 label을 특별토큰으로 만들어 문장의 앞에 넣어줌.
- 데이터나 모델의 출처 표기를 위해 plan파일을 gitignore에서 제외하기로 하였음.

# data
- [챗봇 데이터](https://github.com/songys/Chatbot_data)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- [한국어 대화 요약](https://aihub.or.kr/aidata/30714)
- [한국어 SNS](https://aihub.or.kr/aidata/30718) -> STT를 사용하는 특성상 오타가 다수 포함된 데이터라 사용여부를 고민.
- [멀티모달 영상](https://aihub.or.kr/aidata/137)
- 해당 데이터들을 전부 사용자 대화 앞에 감정 토큰을, 시스템 응답을 백하얀의 페르소나에 맞게 바꿔 사용함. -> 한국어 대화 데이터셋은 상황에 맞지 않아 사용하지 않음.
- 특별 토큰은 차례대로 `"[HAPPY]", "[PANIC]", "[ANGRY]", "[UNSTABLE]", "[HURT]", "[SAD]", "[NEUTRAL]"`.
- 각 데이터를 S1<s>S2<s>, R1</s> 형태로 바꿈(<s> - bos_token, </s> - eos_token) | S1<s>S2<s> or S1<s>S2</s>
- 각 데이터를 columns=["S1", "R1", "S2", "R3", "S3", "R3"]의 형태로 만들어 데이터셋 제작중에 합침
- 추가로 특정 토큰을 가진 질문에 특정 단어가 답변에 많이 포함되게 유도함.
- 질문쪽 각 데이터를 정규화? -> 정규화를 한다면, 오타나 페르소나 오류(존댓말)를 고치거나, konlpy토크나이저로 조사를 없앰 -> 시간(제작시간, 요청시간)의 문제로 보류.

# 모델
- [byeongal/Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 를 파인튜닝해 사용 -> [cc-by-nc-sa-4.0] License(non-commercial)
- [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) 으로의 변경도 고려. -> 데이터가 많으면 kogpt, 아니면 dialoGPT. -> 일단 둘다 코드는 짜둠.
- 캐릭터모델에 필요한 캐릭터 페르소나 탐지기/대화체 변화 모델/대화주제탐지기+대화 씬 탐지모델/프롬프트 인코더 등은 시간과 관련 지식의 부족으로 추후 구현하기로 함.
- 토크나이저 파일을 모두 가져와 특별 토큰들을 추가해 토크나이저로 사용, 허깅페이스 함수의 사용을 위해 저장된 모델을 tf_model.h5 로 다시 저장함.

