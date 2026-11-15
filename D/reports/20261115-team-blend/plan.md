# 계획: blend6에 팀원 모델 추가 (D039)
- 질문: BM25 / LightGCN(고정 epoch) / SimpleX를 blend6 RRF에 넣으면 오르나?
- 방법: blend6 구성(kNN 1, VBPR3 0.7, FREEDOM3 1.3, EASE 1)에 각 모델 가중치 {0.3,0.5,1} 단독 추가, 이후 좋은 것끼리 조합.
- 베이스라인: blend6_e1_r0_i0 0.05744
- 성공 기준: > 0.05794 (+0.0005). 그 이하면 블렌드 이득 없음으로 기록.
- 지표: 키트 score, CPU, 30분.
