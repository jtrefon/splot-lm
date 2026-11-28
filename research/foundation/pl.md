# Założenia projektu — 26.11.2026
Status: notatka robocza, nie publikacja. Wersja referencyjna: [EN](en.md), v0.1.

**Dokument historyczny:** opisuje stan i założenia z chwili powstania, nie bieżące możliwości. Aktualny [stan projektu](../../PROJECT_STATE.md) i [wyniki](../../experiments/results/INDEX.md) obejmują działającą implementację, natywne FP8/Muon, testy CUDA i przygotowany rzeczywisty korpus.

Punktem wyjścia jest edukacyjny [llm-mini](https://github.com/jtrefon/llm-mini): Transformer dekoderowy, RoPE, RMSNorm, SwiGLU, GQA, pamięć KV, strumieniowanie Wikipedii, trening wstępny i SFT. Konfiguracja: 22 warstwy, szerokość 768, 12 głowic Q i 4 KV, FFN 2048, BF16, sekwencja 1024, efektywny batch 64. Przed wykorzystaniem jako model odniesienia trzeba sprawdzić rozłączność danych walidacyjnych.

Proponujemy układ prelude–współdzielony rdzeń–coda. Rdzeń: trzy bloki Gated DeltaNet i jeden GQA, początkowo stała liczba iteracji na sekwencję. MoE: ośmiu ekspertów kierowanych, top-2, jeden współdzielony; SwiGLU, router FP32, regularyzacja obciążenia. MTP od pierwszej generacji; obraz i dźwięk odłożone. LoopMTP i Floodgates stanowią odrębne inspiracje dla nadzoru latentnego i generowania propozycji tokenów. Medusa to model odniesienia, a nie przesądzony wybór.

Hipotezy: lokalność ekspertów między iteracjami zmniejsza transfer pamięci; iteracje mogą wzmacniać błędy kwantyzacji; adaptacyjna głębokość może poprawić kompromis jakości i obliczeń; wspólne projektowanie MTP i MoE może poprawić wydajność bez zmiany rozkładu docelowego.

Sprzęt: RTX 4090 24 GB. Najpierw BF16, następnie pomiary FP8; brak natywnego NVFP4 na Ada. Symulowana kwantyzacja 4-bitowa nie jest natywnym treningiem 4-bitowym. Większe modele: eksperci w RAM, cache na GPU i pomiary PCIe; początkowo wszystkie wagi na GPU. Planowane skale: 30–50M, 150M, 500M, 1–2B, 4–8B, 16B+ (hipotezy skalowania). Pilotaż 100M–1B tokenów, nie pełny trening produkcyjny.

Proporcje danych (hipoteza): 40% edukacyjny web, 20% kod, 15% matematyka, 10% Wikipedia, 10% dokumentacja techniczna, 5% wielojęzyczne z polskim. Wymagane licencje, deduplikacja i odseparowane zbiory testowe. Angielski referencyjny, polskie tłumaczenia. Niezależne badania, bez deklarowania afiliacji PLLuM/NASK. Wyniki, wagi i publikacje dopiero po weryfikacji.
