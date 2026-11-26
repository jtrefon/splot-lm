# Gęsty model referencyjny v0.1
26.11.2026 | #9 | [EN — pełny kontrakt](dense-reference.md)

Pierwszy rzeczywisty trenowalny model dekoderowy SPLOT: GQA, RoPE, RMSNorm, SwiGLU, wspólna macierz embeddingów i wyjścia, maska przyczynowa oraz izolacja dokumentów. To kontrola gęsta, nie docelowa architektura rekurencyjna GDN/MoE/MTP. Konfiguracje JSON: smoke 27 040 parametrów, Dense-Nano 25 172 352, Dense-Mini 75 510 528 (dwa ostatnie sprawdzone wyłącznie jako inwentarz tensorów meta, bez forward). Nie są kontrolami dopasowanymi liczbą parametrów do SPLOT-Nano/Mini.

Wejście: input_ids, document_ids, position_ids, valid o wymiarze [B,T]. Padding nie dopisuje KV. Zmiana dokumentu resetuje pozycje i stan. Cache GQA zawiera obrócone K i nieobrócone V tylko ostatniego dokumentu w każdym wierszu; oddzielny dla warstwy i wiersza. Nie modyfikujemy poprzedniego cache, więc można zachować prefiks do replay. Cache tylko eval + no_grad; trzeba go porzucić po zmianie wag, urządzenia, precyzji lub kolejności sekwencji. CE korzysta z etykiet przesuniętych przez packer, -100 ignorowane, bez ponownego shiftu.

6 testów CPU float64 przeszło. Smoke FP32, 40 kroków AdamW na dwóch stałych syntetycznych przykładach EN/PL: loss 5,587563 -> 2,033330, maksymalna różnica logitów cache 1,192093e-6. To test nadmiernego dopasowania, nie jakości językowej. Brak GPU w środowisku; protokół RTX 4090 w EN. PR #19 pozostaje niezależny, integracja z jego fixture po scaleniu. Kolejny etap: stała głębokość rekurencyjna all-GQA, MTP i weryfikator; później GDN i MoE. #9 nadal otwarte.
