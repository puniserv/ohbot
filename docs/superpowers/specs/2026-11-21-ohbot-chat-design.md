# Ohbot Chat: rozmowa głosowa po polsku przez OpenAI

Data: 2026-11-21
Status: projekt zatwierdzony w rozmowie, czeka na przegląd specyfikacji

## Cel

Ohbot (OhBrain V3.1, RP2040, port szeregowy USB) słucha przez mikrofon komputera i rozmawia po polsku.
Zastosowanie: warsztaty edukacyjne dla dzieci, w zmiennych warunkach (od cichej sali po gwarną klasę).

Core w Pythonie działa na macOS, Linuksie i Windowsie. Aplikacja na Windows to później ten sam core
spakowany do `.exe` (PyInstaller + okno pywebview), bez osobnego kodu aplikacji.

## Decyzje

| Temat | Decyzja |
|---|---|
| Dostawca AI | OpenAI: rozpoznawanie mowy (STT), chat, synteza mowy (TTS), moderacja |
| Architektura silnika | łańcuch etapów (wariant 1); interfejs `Engine` pozwala dodać później silnik Realtime (wariant 3) |
| Modele | ustawiane w `config.toml`; konkretne nazwy dobrane przy implementacji według aktualnej oferty OpenAI |
| Tryby słuchania | push-to-talk, słowo kluczowe, ciągły; przełączane w trakcie działania |
| Persony | pliki `personas/*.md`: `dzieci`, `pokaz`, `domowy`; wybór w konfiguracji i w UI |
| UI | lokalna strona w przeglądarce: napisy rozmowy i widok etapów „jak myśli robot” |
| Zachowanie robota | etap B: kolor oczu wg stanu, mruganie, rozglądanie, usta w rytm głosu; etap C: gesty dobierane przez model |
| Przerywanie | zawsze: klawisz/przycisk „Stop”; opcjonalnie: przerywanie głosem z progiem głośności |
| Diagnostyka | `doctor` w terminalu i przycisk w UI |
| Klucz API | `OPENAI_API_KEY` w `.env`, plik w `.gitignore` |

## Architektura

```
ohbot-chat/
  core/
    events.py         szyna zdarzeń (stan, napisy, etapy, błędy) -> UI i robot
    listen.py         mikrofon + lokalny VAD; tryby: ptt | wake | continuous
    engine.py         interfejs Engine: wypowiedź (audio) -> zdarzenia + audio odpowiedzi
    engine_chain.py   STT -> moderacja -> chat -> moderacja -> TTS
    speak.py          odtwarzanie, obwiednia głośności -> pozycja ust, przerwanie odtwarzania
    robot.py          Ohbot: kolory oczu, mruganie, rozglądanie, usta; tryb bez sprzętu
    doctor.py         diagnostyka krok po kroku
    web.py            FastAPI + WebSocket, serwuje web/
  web/index.html      napisy, widok etapów, przełączniki, Stop, Nowa rozmowa, Diagnostyka
  personas/           dzieci.md, pokaz.md, domowy.md
  config.toml         modele, tryb, persona, kolory stanów, czułość VAD, próg przerywania
  .env                OPENAI_API_KEY
```

Jeden proces, Python 3.11. Elementy nie wołają się nawzajem bezpośrednio: silnik i słuchanie publikują
zdarzenia, a UI i robot je subskrybują.

Biblioteka `ohbot` (PyPI) służy tylko do silników i oczu. Znane pułapki:
- podczas importu sama szuka portu (`init()` wołane na poziomie modułu) i rozgrzewa syntezę mowy;
- `init(portName)` jest zepsute (`checkPort` bierze `p[0]` z napisu, czyli pierwszy znak), więc używamy autowykrywania;
- na macOS autowykrywanie bierze tylko porty z `usb` w nazwie.

Jej synteza mowy nie jest używana, bo głos daje OpenAI TTS.

## Przepływ rozmowy

```
CZEKA (rozgląda się) -> SŁUCHA (zielony) -> ROZPOZNAJE -> MYŚLI (niebieski)
-> MÓWI (biały, usta) -> CZEKA
```

- Słowo kluczowe, wersja pierwsza: tryb ciągły i filtr na tekście z STT. Odpowiedź tylko wtedy,
  gdy wypowiedź zaczyna się od „Ohbot” albo jednego z jego wariantów („hej ohbot”, „ochbot”, „obot”).
  Każda wypowiedź w pokoju trafia wtedy do STT i kosztuje. Później można to zastąpić lokalnym detektorem słowa.
- Historia: ostatnie ok. 10 wymian; „Nowa rozmowa” ją czyści.
- Odpowiedzi krótkie: instrukcja w personie i limit tokenów.
- Gdy robot mówi, mikrofon jest wyciszony, chyba że włączone jest przerywanie głosem.
- Przerywanie głosem: jeśli poziom mikrofonu przekracza próg (suwak w UI) dłużej niż ok. 300 ms,
  robot przestaje mówić i przechodzi w SŁUCHA.
- Moderacja (darmowy endpoint OpenAI) sprawdza pytanie i odpowiedź.

## Błędy

Robot nigdy nie zamiera w milczeniu. Komunikaty awaryjne są generowane przez TTS przy pierwszym uruchomieniu
i zapisywane w pamięci podręcznej, więc działają bez internetu.

| Sytuacja | Reakcja |
|---|---|
| Brak internetu lub timeout | oczy czerwone, „Nie mam połączenia, spróbuj za chwilę”, ponowna próba |
| `insufficient_quota` | komunikat dla prowadzącego w UI: doładuj konto |
| Moderacja zablokowała | „Porozmawiajmy o czymś innym” |
| Pusta transkrypcja | „Nie usłyszałem, powtórz proszę” |
| Robot odłączony | rozmowa trwa w trybie bez sprzętu, stan robota w UI, ponowne połączenie co 5 s |

## Diagnostyka (`doctor`)

Wyniki ✔/✘ z podpowiedzią po polsku:

1. `.env` i klucz obecne
2. Internet i czas odpowiedzi API
3. Klucz ważny (lista modeli)
4. Modele z `config.toml` dostępne
5. Środki: OpenAI nie udostępnia salda przez klucz, więc test to jedno minimalne zapytanie do chatu; `insufficient_quota` oznacza brak środków
6. Mikrofon: lista urządzeń, 2 s nagrania, poziom
7. Głośnik: dźwięk testowy
8. Ohbot: port, odpowiedź firmware (`v`), kolor oczu, kiwnięcie głową

## Prywatność (warsztaty z dziećmi)

Audio i tekst trafiają do OpenAI (STT, moderacja, chat, TTS). Według zasad OpenAI dane z API nie są używane do
trenowania modeli, a logi do wykrywania nadużyć są przechowywane do 30 dni. Prowadzący informuje szkołę lub rodziców
i zbiera zgodę rodziców, gdzie wymaga jej prawo. Tylko push-to-talk ogranicza to, co trafia do OpenAI: w trybie słowa
kluczowego i ciągłej rozmowy każda wypowiedź w sali idzie do rozpoznania mowy, bo słowo „Ohbot” jest sprawdzane
dopiero w transkrypcji. Aplikacja nie zapisuje nagrań na dysk. Szczegóły dla prowadzącego: docs/windows.md, „Privacy”.

## Testy

`pytest`, bez sprzętu i bez internetu:
- maszyna stanów, w tym przerywanie;
- filtr słowa kluczowego;
- obwiednia głośności -> pozycja ust;
- próg przerywania głosem;
- silnik z podstawionym klientem OpenAI: timeout, `insufficient_quota`, blokada moderacji, pusta transkrypcja;
- robot z udawanym portem szeregowym.

Prawdziwe API i sprzęt sprawdza `doctor` oraz lista kontrolna przed warsztatami.

## Uruchamianie

```
uv run ohbotchat           # serwer + otwiera przeglądarkę
uv run ohbotchat doctor    # diagnostyka
```

## Etapy

1. Core w terminalu: push-to-talk -> łańcuch OpenAI -> głos, usta, kolory oczu (B); `doctor`
2. Strona: napisy, widok etapów, przełączniki, Stop, Nowa rozmowa, Diagnostyka
3. Tryby: ciągły, słowo kluczowe, przerywanie głosem z suwakiem
4. Gesty dobierane przez model (C)
5. Aplikacja Windows (`.exe`)
6. Silnik Realtime, jeśli opóźnienie przeszkadza

## Poza zakresem

- Inni dostawcy AI niż OpenAI (interfejs `Engine` zostawia na to miejsce)
- Lokalny detektor słowa kluczowego (później, zamiast filtra na tekście)
- Usuwanie echa w oprogramowaniu; zalecenie sprzętowe: mikrofon konferencyjny z AEC
