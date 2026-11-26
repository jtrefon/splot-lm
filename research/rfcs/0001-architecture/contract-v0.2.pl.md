# Kontrakt architektury SPLOT v0.2
Projekt, 26.11.2026 | #3 | [EN — wersja referencyjna](contract-v0.2.md)

Skrócona wersja polska; pełne równania, kształty i kryteria w EN. Uzgadnia RFC 0002–0005, nie jest wynikiem treningu. Nano ma d=384, 6/2 głowic GQA, 6 głowic GDN, FFN 1024/512; Mini d=768, 12/4 GQA, 12 GDN, FFN 2048/1024. Oba: V=32768 (pojemność, tokenizer do wyboru), głowice 64, splot 4, układ 2/4/2, R_cap=4, R domyślnie 2, osiem ekspertów kierowanych top-2 plus współdzielony. Prelude/coda: GQA i gęste SwiGLU; rdzeń 3 GDN + 1 bramkowane GQA, każde z MoE. Kontrole I0 (bez ponownego wstrzyknięcia), S0 (bez skalowania residual), oddzielne ablacje. Wagi wspólne, stany logiczne osobne. BF16 dopiero po referencji CPU, macierz GDN i redukcje routera FP32.

Natywne MTP: H_r1 to wspólna coda + RMSNorm i macierz E^T; H_r2 to wspólna projekcja W2[d,d], odrębna RMSNorm i E^T. Offset 2 przewiduje token t+2, bez ujawniania etykiet. A0 tylko końcowe NTP, A1 NTP na wszystkich głębokościach, A2 dodatkowa CE offsetu 2 na unikatowych głębokościach {1,R}. Puste maski dają zero, granice dokumentów wykluczone. Wagi 0,25/0,25 oraz balans 0,01 to propozycje pilotażu. Latentne MTP, projekcja płytka, maskowany sufiks i adaptacja później.

| Parametry | Nano | Mini |
|---|---:|---:|
| Bez MTP | 42 907 236 | 146 397 192 |
| Moduł MTP | 147 840 | 590 592 |
| Razem | 43 055 076 | 146 987 784 |

Parametry fizyczne liczone raz, niezależnie od R. Dla jednego tokenu footprint aktywnych parametrów to suma unikatowych wybranych ekspertów po iteracjach oraz parametry niesparse; nie jest to liczba FLOP ani bajtów z GPU. Mini R=1: 89 774 088; R=2: 89 774 088–108 648 456; R=4 do pełnych 146 397 192 (bez MTP). MAC liczy projekcje, splot, kontrakcje GDN, QK/AV, ekspertów i router oraz projekcje słownika; nie operacje skalarne, backward, optimizer ani transfer. Przy T=2048 i R=4 Mini zwykły cel 234 983 424 MAC/token, 20 bloków logicznych; A2 z m=4 odczytami NTP i n=2 offsetu 2 ma 418 615 296 MAC na ostatniej pozycji i 26 bloków. Prefill wszystkich logitów to odpowiednio 455,489 i 812,249 mld MAC. Nie są to pomiary sprzętu. Stan hybrydowy Mini 18,408 MiB na sekwencję przy T=2048,R=4, dodatkowe odczyty coda +12 MiB; snapshot GDN 2,408 MiB. Nano 9,204 MiB. Osobne cache i rollback macierzy, splotu, KV oraz metadanych.

Sześć testów rachunkowych CPU (w tym niezależny spis tensorów PyTorch meta) potwierdza dokładność inwentarza, nie modelu forward. Kolejność #8 dane/tokenizer, #9 gęsta referencja, stała rekurencja i natywne MTP, dokładna weryfikacja, hybryda, MoE, później kwantyzacja i lokalność. Nano przed Mini. Rzeczywiste gradienty, testy całego modelu, pomiary GPU i licencje danych pozostają otwarte.
