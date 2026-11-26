# RFC 0004 — Lokalność ekspertów przy zachowaniu różnorodności iteracji
Projekt v0.1 | #6 | [EN — wersja referencyjna](en.md)

Hipoteza H5: czy polityka rezydencji może wykorzystać przyczynowe wskazówki z iteracji i draftera bez zmiany rozkładu docelowego, a ograniczona regularyzacja lokalności poprawić kompromis jakości i transferu bez utraty zysku z głębokości? Brak deklaracji nowości; DuoMI, LOOM, ReMoE, SP-MoE i CCAMoE są bliskimi pracami.

Router: osiem ekspertów kierowanych, top-2 plus współdzielony; softmax FP32, znormalizowane wagi, deterministyczne remisy, brak odrzucania tokenów we wzorcu. Globalne równoważenie obciążenia, pomiary osobno dla iteracji i domen. Porównujemy router współdzielony, warunkowany głębokością i odrębny per iteracja. Mierzymy przecięcie zestawów top-k dla sąsiednich tokenów i kolejnych iteracji tego samego bloku (oddzielne osie). Nie maksymalizujemy ponownego użycia: niezbędna różnorodność ekspertów.

Faza cache: klucz wagi (checkpoint, blok fizyczny, ekspert, format), bez numeru iteracji dla wag współdzielonych. Model docelowy zachowuje oryginalne top-k, chybienie powoduje załadowanie wagi. Polityki: wszystkie rezydentne, statyczne, LRU, LFU, przyczynowe wskazówki z iteracji/draftera, oracle z wiedzą o przyszłości tylko jako ograniczenie offline. Budżet w bajtach. Wagi niemutowalne mogą pozostać w cache po odrzuceniu propozycji, ale stany KV/GDN/coda muszą być przywrócone. Mierzymy zanieczyszczenie cache, zbędne transfery i opóźnienia.

Faza treningu: odległość d=1-suma minimum prawdopodobieństw ekspertów między iteracjami; proxy nie jest rzeczywistym ruchem bajtów. Kara przedziałowa relu(d_min-d)^2+relu(d-d_max)^2 zamiast wymuszania pełnej zgodności. Osobne kontrole lokalności czasowej i głębokości. Model po treningu ma inny rozkład, więc osobno mierzymy jakość.

Ślady: hash checkpointu, identyfikator uruchomienia/sekwencji, faza, pozycja, iteracja, blok, eksperci, wagi, bajty, kolejność, trafienia, transfer/prefetch, akceptacja, czas; bez tekstu domyślnie. R=1/2/4, budżety 2/4/6 z 8 ekspertów lub równoważne bajty. Oddzielamy symulację od pomiarów wymuszonego offloadu. Proponowany próg prerejestracji: ponad 10% mniej transferu i ponad 10% przyspieszenia end-to-end na ustalonym obciążeniu przy równoważnym celu; trening dodatkowo <=1% względnego wzrostu perplexity oraz tolerancje zadań i zysku z głębokości. Progi do zatwierdzenia przed pomiarami. Wyniki negatywne zachowujemy.

Zależności #2–#5, #8–#10. Pełne teksty priorytetowych prac pozostają do przeglądu. Brak implementacji i pomiarów.
