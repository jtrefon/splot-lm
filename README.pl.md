# SPLOT — Sparse Predictive Looped Optimized Transformer

[English](README.md) | [Architektura](research/rfcs/0001-architecture/pl.md) | [Plan prac](ROADMAP.md)

SPLOT to niezależna inicjatywa badawcza o polskich korzeniach, otwarta na międzynarodową współpracę. Badamy efektywne architektury modeli językowych: rekurencyjną głębokość obliczeniową, hybrydowe mechanizmy uwagi, rzadką aktywację ekspertów (MoE), natywną predykcję wielotokenową (MTP), trening z obniżoną precyzją i inferencję uwzględniającą ograniczenia sprzętowe.

**Status: propozycja badawcza, przed implementacją.** Nie opublikowano jeszcze wytrenowanego modelu ani wyników eksperymentalnych. Projekt nie jest afiliowany przy PLLuM ani NASK.

Pytanie badawcze: czy model może uczyć się doboru aktywnych parametrów, głębokości obliczeń i reprezentacji kontekstu, ograniczając transfer pamięci i błędy numeryczne, aby poprawić kompromis między jakością, opóźnieniem i zużyciem pamięci?

- [Założenia projektu](research/foundation/pl.md)
- [RFC 0001](research/rfcs/0001-architecture/pl.md)
- [RFC 0005: MTP](research/rfcs/0005-native-mtp/pl.md)
- [Literatura](research/literature/README.md)
- [Zasady współpracy](CONTRIBUTING.md) i [licencje](LICENSING.md)

Angielski jest językiem referencyjnym specyfikacji i kodu. Zapraszamy do dyskusji i zgłaszania zmian po polsku lub angielsku.
