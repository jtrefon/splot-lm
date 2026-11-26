# SPLOT — sprzętowo świadomy rekurencyjny, rzadki i predykcyjny model językowy
Roboczy whitepaper v0.1, 26.11.2026 | [EN](architecture-v0.1-en.md)

## Streszczenie
Proponujemy architekturę dekoderową łączącą współdzieloną głębokość rekurencyjną, hybrydę Gated DeltaNet/GQA, MoE i natywne MTP. Badamy kompromis jakości, obliczeń i pamięci na sprzęcie konsumenckim. Warianty Nano 43,06M i Mini 146,99M parametrów wynikają z analitycznego inwentarza potwierdzonego tensorami meta, nie z wytrenowanych modeli. Dokument jest propozycją badawczą, nie publikacją wyników.

## Metoda
Osiem fizycznych bloków, układ 2/4/2, rdzeń 3 GDN + 1 bramkowane GQA i MoE. Stała liczba iteracji 1–4, odrębne stany logiczne, współdzielone wagi i MTP offsetu 2. Dokładna weryfikacja spekulacyjna zachowuje rozkład docelowy; cache wag nie może zmieniać routingu docelowego. Osobne eksperymenty routingu trenowanego. Kontrole gęste, rekurencyjne, hybrydowe i NTP/MTP. Szczegóły: [kontrakt](../rfcs/0001-architecture/contract-v0.2.pl.md).

## Ocena i ograniczenia
Prerejestracja danych, podziału, tokenizera, budżetu, seedów i sprzętu. Pomiary loss, rozumowania, kodu, retrieval, różnorodności ekspertów, VRAM, transferów, latencji i rollbacku. Najpierw Nano, potem Mini po walidacji. Dotychczas tylko testy równań, semantyki i inwentarza. Brak wyników treningu, twierdzeń o nowości lub afiliacji. Wymagany przegląd literatury, licencji i pełny model. Wersja polska jest skrótem angielskiej.
