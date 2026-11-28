# SPLOT — Sparse Predictive Looped Optimized Transformer

[English](README.md) · [Karta badawcza](research/CHARTER.pl.md) · [Kontrakt architektury](research/rfcs/0001-architecture/contract-v0.2.pl.md) · [Stan projektu](PROJECT_STATE.md) · [Wyniki](experiments/results/INDEX.md) · [Plan badań](ROADMAP.md)

**Osiągać więcej dzięki lepszej architekturze, nie większej skali.**

Niezależny, otwarty program badań nad architekturami modeli językowych, ukierunkowany na lepszą jakość przy równoważnym lub niższym koszcie obliczeniowym. Sprzęt konsumencki jest początkowym warunkiem walidacji, nie celem badań. Zob. [kartę założycielską](research/CHARTER.pl.md) i [ramy oceny](research/evaluation/README.md) (EN).

SPLOT łączy rekurencyjną głębokość obliczeń, hybrydę Gated DeltaNet / grouped-query attention (GDN/GQA), rzadką aktywację ekspertów (MoE), natywną predykcję wielotokenową (MTP) oraz trening z obniżoną precyzją. Badamy, czy ich połączenie poprawia kompromis między jakością, kosztem obliczeń, opóźnieniem i zużyciem pamięci.

**Stan na 28 listopada 2026: działająca architektura badawcza, zweryfikowany natywny trening CUDA; zakończono ograniczony pilotaż wykonalności na rzeczywistych danych.** Implementacja, testy CPU/GPU i wersjonowane wyniki są dostępne publicznie. Nie wydano produkcyjnego modelu ani publicznych wag po pretreningu. Projekt nie jest afiliowany przy PLLuM, NASK ani innej instytucji.

## Architektura

- Osiem fizycznych bloków w układzie 2/4/2: dwa GQA, współdzielony rdzeń 3 GDN + 1 bramkowane GQA z MoE, dwa GQA. Stała głębokość 1–4, oddzielne stany przyczynowe dla logicznych wywołań.
- Ośmiu ekspertów kierowanych, top-2 i jeden współdzielony; router FP32 i regularyzacja obciążenia.
- Natywne MTP offsetu 2 oraz referencyjna dokładna weryfikacja spekulacyjna D1/D2 z zaakceptowanym baseline D0 i śladem v2 ([T0018](experiments/results/T0018/attempt-002/summary.md)). Pomiary akceptacji i latencji na trenowanych wagach pozostają do wykonania.
- Natywne FP8 (E4M3 dla wag i aktywacji, E5M2 w propagacji wstecznej), BF16 w strumieniu rezydualnym, FP32 dla wag nadrzędnych, stanów wrażliwych i optymalizatora. Kontrole FP32/BF16.
- Hybrydowy Muon/AdamW oraz opcjonalne ponowne wykorzystanie skwantowanych wag FP8 w ramach jednej aktualizacji.

| Wariant | Parametry fizyczne (słownik modelu 32 768) | Stan |
|---|---:|---|
| Nano | 43 055 076 | Implementacja, trening syntetyczny i testy CUDA długich sekwencji |
| Mini | 146 987 784 | Konfiguracja i inwentarz parametrów; dłuższy trening GPU niezweryfikowany |

## Wybrane wyniki

Pomiary sprzętowe dotyczą referencji na jednej karcie RTX 4090 24 GB, PyTorch 2.10.0/CUDA 12.8, a nie produkcyjnej przepustowości.

| Obserwacja | Wynik | Dowody |
|---|---|---|
| Natywne GEMM FP8 | 33/33 przypadki forward, dgrad i wgrad | [T0007](experiments/results/T0007/attempt-001/summary.md) |
| Autograd FP8 | 156 projekcji Nano, testy R1/R2/R4 i dokładne wznowienie | [T0009](experiments/results/T0009/attempt-002/summary.md) |
| FP8 + Muon | 35 kroków syntetycznych; NLL 1,4900 vs 3,4992 FP8/AdamW i 1,6966 BF16/Muon | [T0011](experiments/results/T0011/attempt-001/summary.md) |
| Sekwencja 4096 | R2/B1: FP8/Muon 14,62 GiB szczytowej rezerwacji, 10,136 s; BF16/Muon 14,91 GiB, 9,804 s | [T0012](experiments/results/T0012/attempt-001/summary.md) |
| Współdzielenie wag FP8 | T512: 298 → 156 kwantyzacji, identyczny stan modelu/optymalizatora i loss; 1,949 → 1,633 s w jednym pomiarze | [T0017](experiments/results/T0017/attempt-001/t512-pair.json) |
| Korpus PL/EN | 8 rodzin utworów, 8,57 MiB tekstu, BPE 16 384, 56 201 wybranych tokenów treści | [T0016](experiments/results/T0016/attempt-001/summary.md) |

Wyniki syntetyczne z jednym seedem nie dowodzą generalizacji ani przewagi optymalizatora. Pojedynczy pomiar czasu nie dowodzi trwałego przyspieszenia. Korpus przygotowano na Macu; [Pilotaż na Linuksie](experiments/results/T0017/attempt-002/summary.md) zakończył 64 aktualizacje; NLL walidacji 7,744022, 45,1% trafień cache kwantyzacji. Zaakceptowano wykonalność z [udokumentowanymi odstępstwami proceduralnymi](experiments/results/T0017/review-supplement-001/summary.md), nie przewagę modelu ani pełną zgodność proceduralną. Checkpoint zachowano w prywatnej trwałej pamięci lokalnej. T0017 ma status blocked w pierwszej próbie z powodu niewykonanego pilotażu, mimo akceptacji niezależnej implementacji. Surowe dane i tensory wag pozostają lokalne.

## Współpraca i reprodukcja

W kompatybilnym środowisku PyTorch, z czystego checkoutu: `PYTHONPATH=src:tools python tools/ci_cpu.py`. CI CPU weryfikuje regresje i integralność dowodów, nie odtwarza eksperymentów GPU. Dalsze wykonanie zgodnie z [REP](EXECUTION.md).

[Dokumentacja PL](docs/pl/README.md) · [Literatura](research/literature/README.md) · [Hipotezy](research/hypotheses/README.md) · [Zgłoszenia](https://github.com/jtrefon/splot-lm/issues) · [Licencje](LICENSING.md) · [Cytowanie](CITATION.cff)

Angielski pozostaje językiem referencyjnym. Kod Apache-2.0; licencje tekstów badawczych i przyszłych wag wymagają oddzielnych decyzji. Zapraszamy do niezależnej replikacji i merytorycznej współpracy.
