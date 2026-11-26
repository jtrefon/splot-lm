# Natywne MTP i samospekulacja — ukierunkowany przegląd literatury
26.11.2026 | Projekt v0.2 | Zgłoszenie #2 | [EN — wersja referencyjna](en.md)

Pytanie: które cele treningowe i mechanizmy inferencji zastosować w SPLOT przy współdzielonym rdzeniu GDN/GQA, MoE i pojedynczej RTX 4090? Celem jest dokładne dekodowanie zgodne z rozkładem modelu pełnej głębokości oraz efektywność obliczeń. Jest to przegląd narracyjny, nie systematyczny. Przeanalizowano wybrane rozdziały pełnych tekstów Floodgates, LoopSpec, LoopMTP i DeepSeek-V3; dla Medusa, Gloeckle, EAGLE-3, Leviathan i LoopFill wykorzystano streszczenia lub strony wydawnicze. Pełny tekst HTML LoopFill nie był dostępny. Nie przeprowadzono replikacji.

| Praca | Mechanizm | Rola |
|---|---|---|
| [Leviathan i in.](https://proceedings.mlr.press/v202/leviathan23a.html) | Dokładne próbkowanie spekulacyjne | Wzorzec poprawności |
| [Gloeckle i in.](https://proceedings.mlr.press/v235/gloeckle24a.html) | Niezależne głowice przyszłych tokenów | Kontrola MTP |
| [Medusa](https://proceedings.mlr.press/v235/cai24b.html) | Głowice równoległe, drzewo propozycji | Kontrola; trudniejsza weryfikacja GDN |
| [DeepSeek-V3](https://arxiv.org/html/2412.19437) | Sekwencyjne moduły MTP | Kontrola warunkowana |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | Autoregresyjny drafter korzystający z cech modelu | Porównanie później |
| [LoopMTP](https://arxiv.org/html/2608.03624v1) | Dopasowanie kosinusowe stanów do embeddingów przyszłych tokenów | Tani cel latentny, nie dekoder |
| [LoopSpec](https://arxiv.org/html/2609.17184v1) | Płytkie propozycje, potokowanie, bez dodatkowego treningu | Punkt odniesienia |
| [Floodgates](https://arxiv.org/html/2611.33896v1) | Adaptacja płytkiej iteracji, maskowany sufiks i głęboki cache prefiksu | Zaawansowany kandydat |
| [LoopFill](https://arxiv.org/abs/2611.19649) | Sufiks z wyszukiwania i płytkie propozycje | Pełny tekst do sprawdzenia |

Wnioski: nadzór latentny nie jest równoważny rozkładowi propozycji; szybki drafter nie gwarantuje poprawy jakości treningu. Floodgates korzysta z nadzoru na wielu głębokościach i adaptacji do głębokiego cache. Nie wolno po prostu zamieniać stanów GDN lub KV. Wyniki autorów (do 7,5x Floodgates, do 6,83x LoopSpec; średnie 5,85x i 5,13x LoopFill na różnych modelach) nie są wynikami SPLOT ani porównaniem w identycznych warunkach.

Decyzja wstępna: od pierwszej generacji wspólny odczyt NTP na wielu głębokościach oraz cel predykcji dodatkowego tokenu, z osobnymi kontrolami NTP i nadzoru na wszystkich głębokościach. Cel latentny testujemy niezależnie. Najpierw dokładna samospekulacja sekwencyjna; następnie adaptowane sloty maskowane, K=2. Głęboki cache draftera ma być tylko do odczytu, a pełny weryfikator pozostaje niezmieniony. Badać akceptację, jakość, FLOP treningu, koszt odtwarzania stanu, ruch ekspertów i całkowitą latencję. Nie deklarujemy nowości bez przeglądu wcześniejszych prac.

Otwarte: pełny tekst LoopFill, szczegóły Medusa, EAGLE-3 i efektywnej pamięciowo straty Gloeckle, szerszy przegląd bieżącej literatury i licencji. Zob. [protokół i decyzja techniczna (EN)](../../rfcs/0005-native-mtp/decision-v0.2.md).
