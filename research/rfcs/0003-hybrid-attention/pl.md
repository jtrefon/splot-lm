# RFC 0003 — Hybrydowe Gated DeltaNet i GQA
Projekt v0.1 | 26.11.2026 | #5 | [EN — wersja referencyjna](en.md)

Polski skrót; pełne równania, ABI, obliczenia parametrów i kryteria w EN. Propozycja: dwa niezależne bloki GQA w prelude, rdzeń współdzielony z 3 GDN + 1 bramkowanym pełnym GQA, dwa GQA w coda. Proporcja 3:1 dotyczy rdzenia, nie całego grafu. Porównamy kontrolę all-GQA. GDN v1 jako punkt odniesienia; GDN-2 i Integral DeltaNet jako późniejsze ablacje, nie zmiana bazowej architektury.

Mini: d=768; GDN 12 głowic, d_k=d_v=64, splot przyczynowy głębokościowy K=4 na surowych q/k/v (2304 kanały). q,k normalizowane L2, q dodatkowo skalowane przez 1/sqrt(d_k), v bez tej normalizacji. Z i bramki nie przechodzą przez splot. beta=sigmoid(xW_b), g=-exp(log_A)*softplus(xW_a+dt_bias), alpha=exp(g). Stan S ma kształt [B,12,64,64], historia surowych projekcji [B,3,2304]. D_t=alpha_t S_(t-1), u_t=beta_t(v_t-k_t^T D_t), S_t=D_t+k_t u_t^T, o_t=q_t^T S_t. Najpierw decay, potem odczyt błędu, zapis i odczyt nowego stanu. Na głowicę RMSNorm i bramka SiLU(z), projekcja wyjścia do d. Stan FP32, historia BF16 w inferencji BF16.

Niezależny algorytm blokowy: G_i=sum g_j, gamma_i=exp(G_i), delta_ij=exp(G_i-G_j) dla i>=j. S_i=gamma_i S_0+sum delta_ij k_j u_j^T. Rozwiązujemy dolny układ trójkątny z jedynkami na przekątnej i L_ij=beta_i delta_ij k_i^T k_j; RHS=beta_i(v_i^T-gamma_i k_i^T S_0). Odczyty po aktualizacji. Różnice logarytmów zamiast dzielenia iloczynów. Dokładne zero alpha obsługujemy sekwencyjnie lub dzielimy segment; granice dokumentów oddzielne. To niezależny oracle algebraiczny, nie zoptymalizowane WY.

GQA: Q 12x64, KV 4x64, grupa 3. RoPE na Q/K w pozycji dokumentu, theta=1e6, V bez rotacji, pełna maska przyczynowa. Cache przechowuje tylko cztery głowice KV. W rdzeniu bramka sigmoid z wejścia po normalizacji przed projekcją wyjściową; prelude/coda bez bramki. Brak normalizacji QK w referencji. Padding nie aktualizuje żadnego stanu, nowy dokument zeruje macierz, historię splotu i KV. Każda iteracja i odczyt coda mają osobny stan.

Rollback: kopia S i surowej historii, pozycje, długości i logity prefiksu; stan roboczy weryfikatora oddzielny. Po odrzuceniu przywrócenie i replay zaakceptowanych tokenów z korektą dokładnie raz. Samo obcięcie KV nie odtwarza GDN; nie zakładamy odwracalności macierzy. Cache wag ekspertów osobny. Dodatkowe odczyty coda kosztują pamięć.

Parametry mixerów fizycznych: GDN 2 976 856, GQA 1 572 864, bramkowane GQA 2 162 688. Suma mixerów 17 384 712, wstępna suma modelu przed dodatkowymi modułami MTP 146 397 192 (do uzgodnienia w #3). Stan GDN 210 432 B/wywołanie; KV GQA 1024 B/token/wywołanie. R=4, T=2048: 18,408 MiB hybryda vs 40 MiB all-GQA przy tylko końcowym odczycie coda. Trzy dodatkowe odczyty coda dodają 12 MiB. Przy krótkim kontekście GDN może kosztować więcej. To szacunki cache, nie pomiary VRAM ani wydajności.

Sześć testów NumPy float64 przeszło lokalnie: porównanie scan i układu trójkątnego, nieregularne chunki, bramki graniczne, rollback macierzy, historia splotu i reset, grupowane RoPE/GQA. Brak testów całego modelu, gradientów, batch/packing i GPU. #5 pozostaje otwarte. Następnie #3 i #8/#9.
