# Lokalność ekspertów, rekurencyjna głębokość i weryfikacja spekulacyjna
Przegląd v0.1 | 26.11.2026 | #6 | [EN — wersja referencyjna](en.md)

Skrót przeglądu narracyjnego, nie systematycznego. Sprawdzono streszczenia/strony wydawnicze oraz wybrane sekcje HTML. Nie wykonano replikacji. Szczegóły zakresu i odsyłacze w EN.

- [Not All Models Suit Expert Offloading](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6a7a2238da2b755e6dc1dbe278c60c2a-Abstract-Conference.html): lokalne równoważenie obciążenia i lokalność mogą pozostawać w konflikcie.
- [ReMoE](https://proceedings.mlr.press/v306/zhu26ab.html): dostrajanie routera dla ponownego użycia ekspertów przez kolejne tokeny.
- [Cacheable by Design?](https://arxiv.org/abs/2608.18261): przy modelu 137M zmniejszono liczbę chybień cache, ale nie spełniono prerejestrowanego progu wzrostu perplexity <=1%; większy model nie rozwiązał kompromisu.
- [Cache-Aware Joint Router Adaptation](https://arxiv.org/abs/2609.04895): osobne routery cache, zachowanie natywnego top-k.
- [SP-MoE](https://arxiv.org/abs/2510.10302): prefetch ekspertów na podstawie draftera.
- [CCAMoE](https://arxiv.org/abs/2611.28550): bliski kierunek wykorzystania ekspertów w cache do szkicowania; pełny tekst do analizy.
- [DuoMI](https://arxiv.org/abs/2611.33952): bezpośredni wcześniejszy kierunek regularyzacji ponownego użycia ekspertów między iteracjami, z kontrolą zakresu zmian. Nie możemy deklarować nowości samej lokalności między iteracjami.
- [LOOM](https://arxiv.org/abs/2610.01153): zapadanie się do identycznych wyborów ekspertów może ograniczać korzyść z kolejnych iteracji; odrębne routery zwiększają różnorodność.

Wniosek: badamy węższy problem wspólnego zarządzania rezydencją wag, różnorodnością iteracji i weryfikacją spekulacyjną przy ograniczonym VRAM. Najpierw nie zmieniamy funkcji modelu docelowego, tylko harmonogram i prefetch. Zmieniony router docelowy to osobny eksperyment jakościowy. Wagi ekspertów są niemutowalne; stany KV/GDN wymagają transakcyjnego rollbacku. Mały model w całości na GPU nie dowodzi korzyści PCIe: najpierw symulacja śladów, później pomiar wymuszonego offloadu z ustalonym budżetem bajtów.

DuoMI, CCAMoE i LOOM wymagają pełnego przeglądu; Sticky Routing i CECAR nie zostały zweryfikowane w źródle pierwotnym. Brak twierdzeń o nowości i pomiarów SPLOT. Zob. [RFC 0004](../../rfcs/0004-expert-routing/pl.md).
