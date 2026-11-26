# RFC 0001 — Podstawy SPLOT
Status: projekt v0.1 | 26.11.2026 | [EN — wersja referencyjna](en.md)

Cel: pomiar kompromisu jakości, pamięci, obliczeń, energii i opóźnienia. MTP w pierwszej generacji; multimodalność, adaptacyjna głębokość na poziomie tokenu i duże modele odłożone do czasu potwierdzenia poprawności.

Dla wejścia x o wymiarze [B,T] embedding E ma wymiar [V,d], z=P(E[x]), h_0=z, h_(r+1)=C(h_r+a_r z). P to dwa bloki wstępne, C to cztery współdzielone bloki, a_r to konfigurowalna skala wstrzykiwania wejścia (0 w wariancie odniesienia). Dwa unikatowe bloki końcowe i współdzielona macierz embeddingów dają logity. Osiem bloków fizycznych, 4+4R wykonań logicznych, R=1..4. Początkowo stałe R na sekwencję.

Rdzeń: 3 Gated DeltaNet i 1 GQA. Rekurencja po pozycjach tokenów jest odrębna od rekurencji po głębokości. d=768, 12 głowic Q, 4 KV, wymiar 64, RoPE theta 1e6. Parametry stanu GDN wymagają doprecyzowania. Pełne GQA przed eksperymentami z uwagą rzadką. MoE: osiem ekspertów kierowanych i jeden współdzielony; top-2, normalizacja wag wybranych ekspertów, SwiGLU, router FP32. FFN eksperta 1024, gęsty 2048. Monitorujemy entropię, obciążenie i nieaktywnych ekspertów. Aktywne 3/9 FFN nie oznacza 1/3 FLOP całego modelu.

Przy V=32768 i d=768 embedding ma 25 165 824 parametrów, eksperci 84 934 656, cztery gęste FFN 18 874 368: razem 128 974 848 przed uwagą, routerami, normami i MTP. ~150M to szacunek, nie pomiar. Raportujemy parametry fizyczne, aktywowane na iterację i koszt wszystkich wykonań logicznych.

Cache jest indeksowany przez blok, iterację i sekwencję. Stany GDN i KV są odrębne dla każdej iteracji; odrzucenie propozycji wymaga odtworzenia stanu GDN, nie tylko obcięcia KV. Testy: zgodność pełnego i przyrostowego forward, izolacja dokumentów, gradienty współdzielonych wag, odtwarzalność, rollback i poprawność rozkładu spekulacyjnego.

Funkcja straty: L_NTP + lambda_m L_MTP + lambda_b L_balance + lambda_l L_locality + lambda_c L_compute. Początkowo ostatnie dwa współczynniki zerowe. Proponowana miara zmiany routingu: 1 minus suma minimum prawdopodobieństw ekspertów w sąsiednich iteracjach; wymaga konfrontacji z rzeczywistym cache. BF16 najpierw, FP8 później, symulowane FP4 QAT po ustaleniu poprawności. Natywne NVFP4 nie na RTX 4090.

Kolejność: gęsty model odniesienia, rekurencja i natywne MTP, hybryda oraz MoE niezależnie i wspólnie, dokładna weryfikacja spekulacyjna, następnie lokalność, adaptacja i kwantyzacja. Otwarte: szczegóły GDN, MTP, normalizacji, tokenizera, praw do danych i analiza literatury routingu. Brak wyników treningu.


Proponowana aktualizacja: [kontrakt architektury v0.2](contract-v0.2.pl.md).
