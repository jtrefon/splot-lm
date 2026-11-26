# RFC 0005 — Natywne MTP i samospekulacja
Projekt v0.1 | [EN — wersja referencyjna](en.md)

MTP od pierwszej generacji. Rozróżniamy nadzór predykcji przyszłych tokenów, nadzór latentny, generowanie propozycji i dokładną weryfikację. Sam nadzór latentny nie gwarantuje dobrego draftera.

Medusa (równoległe głowice) to punkt odniesienia; DeepSeek MTP (sekwencyjne warunkowanie) to alternatywa treningowa; EAGLE-3 to zewnętrzny drafter. LoopMTP wykorzystuje dopasowanie stanów pośrednich do przyszłych embeddingów. LoopSpec wykorzystuje płytkie iteracje bez dodatkowego treningu. Floodgates proponuje równoległe przewidywanie maskowanego sufiksu z płytkich iteracji. LoopFill wymaga weryfikacji źródła.

Kolejność: M0 końcowy NTP; M1 nadzór CE dla stanów pośrednich i horyzontu początkowo K=2; M2 sekwencyjne MTP z warunkowaniem wyłącznie w głowicy pomocniczej; M3 nadzór latentny z kontrolą gradientu; M4 maskowany sufiks bez ujawniania przyszłych tokenów; M5 pełnogłębokościowa weryfikacja. Maskujemy padding i granice dokumentów. W próbkowaniu akceptacja min(1,p(y)/q(y)), przy odrzuceniu próbkujemy z dodatniej części p-q; równoległe propozycje wymagają poprawnych warunkowych rozkładów lub sprawdzonego algorytmu. W trybie greedy zgodność z argmax. Po odrzuceniu przywracamy wszystkie stany KV/GDN.

Metryki: akceptacja na pozycję, liczba zaakceptowanych tokenów na weryfikację, opóźnienie draftera i weryfikatora, koszt rollbacku, tok/s, TTFT, energia/token, VRAM, jakość i zgodność rozkładu. Porównania przy takim samym kontekście, batchu i ustawieniach dekodowania. Raportowane wyniki innych autorów (Floodgates 7,5x, LoopSpec 6,83x, LoopMTP do 8,1% względnej poprawy) nie są wynikami SPLOT. Przegląd pełnych tekstów pozostaje otwarty.
