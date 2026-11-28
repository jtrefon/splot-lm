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
| Powtarzany pomiar kosztu referencji | T512 BF16/FP8 mediana 1,292/1,600 s; T2048 4,832/5,167 s, po trzy pary; cache wag wyłączony | [T0020](experiments/results/T0020/attempt-001/summary.md) |

Wyniki syntetyczne z jednym seedem nie dowodzą generalizacji ani przewagi optymalizatora. Pojedynczy pomiar czasu nie dowodzi trwałego przyspieszenia. Korpus przygotowano na Macu; [Pilotaż na Linuksie](experiments/results/T0017/attempt-002/summary.md) zakończył 64 aktualizacje; NLL walidacji 7,744022, 45,1% trafień cache kwantyzacji. Zaakceptowano wykonalność z [udokumentowanymi odstępstwami proceduralnymi](experiments/results/T0017/review-supplement-001/summary.md), nie przewagę modelu ani pełną zgodność proceduralną. Checkpoint zachowano w prywatnej trwałej pamięci lokalnej. T0017 ma status blocked w pierwszej próbie z powodu niewykonanego pilotażu, mimo akceptacji niezależnej implementacji. Surowe dane i tensory wag pozostają lokalne.

## Współpraca i reprodukcja

W kompatybilnym środowisku PyTorch, z czystego checkoutu: `PYTHONPATH=src:tools python tools/ci_cpu.py`. CI CPU weryfikuje regresje i integralność dowodów, nie odtwarza eksperymentów GPU. Dalsze wykonanie zgodnie z [REP](EXECUTION.md).

[Dokumentacja PL](docs/pl/README.md) · [Literatura](research/literature/README.md) · [Hipotezy](research/hypotheses/README.md) · [Zgłoszenia](https://github.com/jtrefon/splot-lm/issues) · [Licencje](LICENSING.md) · [Cytowanie](CITATION.cff)

Angielski pozostaje językiem referencyjnym. Kod Apache-2.0; licencje tekstów badawczych i przyszłych wag wymagają oddzielnych decyzji. Zapraszamy do niezależnej replikacji i merytorycznej współpracy.

**Stan ewaluacji:** zintegrowano [gęsty model porównawczy NTP](research/evaluation/T0021-acceptance.md) oraz [profilowanie backward z kontrolą zgodności numerycznej](research/evaluation/T0022-acceptance.md). [Integracja pilotażu ze słownikiem rzeczywistym](research/evaluation/T0023-acceptance.md) przeszła testy CLI na hostowanym CPU po dwóch błędach asercji (zachowano oba nieudane podejścia); następnie ukończono pierwszy porównawczy pilotaż CUDA. [Izolowany pomiar GDN](research/evaluation/T0024-acceptance.md): chunk64 skrócił medianę forward+backward z 0,174688 do 0,011728 s dla T512 i z 0,734590 do 0,046498 s dla T2048 (RTX 4090, syntetyczne projekcje FP32, trzy powtórzenia). [Kwalifikacja pełnego modelu](research/evaluation/T0025-disposition.md) nie przeszła bramek tensorów i routingu BF16 T512 (0,8301% wszystkich decyzji routingu różniło się); bez pomiaru zasobów pełnego modelu ani wdrożenia. Scan pozostaje bez zmian.

**Pierwszy pilotaż porównawczy referencji:** [T0027](research/evaluation/T0027-acceptance.md), RTX 4090, V16384, 3 warianty × 3 ziarna ×128 aktualizacji, 46 076 etykiet NTP/przebieg. Średnie NLL walidacji SPLOT / gęsty model o zbliżonej liczbie parametrów / gęsty model o zbliżonym forward MAC: 7,4141 / 7,4719 / 7,5136. Średnie sumy czasów aktualizacji 127,094 / 5,315 / 5,359 s; referencja SPLOT jest około 24× wolniejsza. Bez wniosku o przewadze jakościowo-kosztowej, populacji ani skalowaniu. [Recenzja projektu protokołu](research/evaluation/T0026-acceptance.md).

**Bieżące badania implementacyjne:** [prototyp v2 pokrycia bajtowego](research/evaluation/T0028-acceptance.md) dotyczy wyłącznie danych syntetycznych; [lokalizacja rozbieżności GDN](research/evaluation/T0029-acceptance.md) odtworzyła różnice routingu BF16 (34/4096), natomiast pełny przebieg FP32 na tym samym syntetycznym przypadku nie zmienił zbiorów ekspertów. Wymuszone trasy zmniejszyły, lecz nie wyeliminowały rozbieżności; bez wniosku o szybkości, uczeniu ani wdrożeniu.

**Najnowsze badania:** [T0030](research/evaluation/T0030-acceptance.md) wzmocniło syntetyczny prototyp audytu bajtów. [T0031](research/evaluation/T0031-acceptance.md) porównało FP32 scan/chunk64 na RTX 4090, Nano R2 A2, syntetyczne T512/T2048: mediany forward+loss+backward 1,130492/0,137413 s i 4,631050/0,379827 s (8,23x/12,19x); trzy przypadki numeryczne przeszły, bez różnic routingu. Selektywne wyspy precyzji odrzucono. Bez wniosku o aktualizacji optymalizatora, uczeniu ani wdrożeniu; domyślny scan bez zmian.

**Najnowsze badania:** [T0032](research/evaluation/T0032-acceptance.md) syntetyczny test stream-v1 (1,012 mln ważnych pozycji; niezależny audyt, ograniczenia strumieniowego I/O pozostają). [T0033](research/evaluation/T0033-acceptance.md) opcjonalny FP32 chunk64: Nano R2 A2 RTX 4090, syntetyczne mediany pełnej aktualizacji T512 1,253690/0,218430 s i T2048 4,794990/0,452887 s (scan/chunk, 5,74x/10,59x). Domyślny scan i natywny FP8 scan pozostają. Bez wniosku o przewadze jakościowej. [RFC QAT w pretreningu](research/precision/qat-pretraining.md) — propozycja, nie wdrożenie.
