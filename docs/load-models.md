# Load models

LTM100 separates the **scenario** (what ops to emit and on what data) from
the **load model** (when and how many users emit them). The same scenario
plan runs under either load model — only the consume schedule changes. This
document explains the two models, the congestion policy, and when to use
each. For per-scenario data flow, see [`scenarios.md`](./scenarios.md).

## The two axes

- **Scenario** — owns the *op mix* (add vs search) and the *data* (which
  memory content to add, which query to search). It yields an infinite
  `plan()` of `Op`s.
- **Load model** — owns the *consume schedule*: who consumes the plan, when,
  and for how long. The runner drives this; it never touches the op mix or
  data.

Because the axes are separate, every scenario runs under both `--model
closed` and `--model open`.

## Closed model (`--model closed`, default)

A fixed pool of `--users N` virtual users, each an asyncio coroutine that
loops the scenario plan back-to-back:

- **Per-user in-flight = 1.** A user sends one request, *waits for the
  response*, then sends the next. A single user never has more than one
  request in flight.
- **Concurrency = min(N, `--global-concurrency`)** (N if no cap). Because
  each user blocks on its response, a slow server makes all N stall together
  and a fast one lets all N run — response time feeds back into concurrency
  (a *closed* loop).
- **`N` is the concurrent user count.** `--users 20` models "20 users always
  connected".
- **Think time** (`Op.delay`) is the only timing variation. `delay=0`
  (add-load) is back-to-back; `delay=uniform(0, 0.05)` (chat-replay,
  mixed) drifts users out of lockstep.

Closed models the classic load test: "with N users hammering the server,
how fast is it?" Use it to measure raw throughput and latency at a fixed
concurrency.

## Open model (`--model open`)

Users *arrive* over time and each runs a short session then leaves:

- **Arrival process (runner-owned).** A Poisson process spawns sessions at
  `--arrival-rate λ`. Inter-arrival is `expovariate(λ)` — bursty but with a
  fixed average rate. Sessions keep arriving until `--duration`.
- **Session lifetime is bounded.** Each arriving session consumes up to
  `--session-ops` ops from the scenario plan, then the user leaves. Users do
  not loop forever.
- **Arrival is the input; concurrency is the output.** `λ` fixes how fast
  sessions arrive; *how many run at once* is whatever the server's service
  rate produces. A fast server keeps concurrency low; a slow one lets it
  pile up. This is an *open* loop — arrival is independent of service rate.
- **Tenancy pool.** Arriving sessions draw users round-robin from the
  `--users N` pool (so the tenant already exists). The same user may appear
  in several concurrent sessions (same tenant; isolation is preserved).
- **`--duration` is required** (open runs are always time-bounded).

Open models real traffic: visitors arriving independently (web site,
messaging app) and each doing a short burst of work.

## Congestion policy (the open model's key output)

Under closed, concurrency is fixed at N so nothing is ever rejected. Under
open, if arrivals outrun the server, requests pile up — and LTM100 must
decide what to do. The policy:

- `--global-concurrency C`: max requests in flight at once.
- `--queue-bound Q`: how many requests beyond `C` may wait for a slot.
- When in-flight reaches `C + Q`, the next request is **rejected**:
  - `status = "rejected"`, `error_kind = "queue_full"`, zero latency
    (it is recorded but not executed, so the rejection rate is measurable).
- `Q=0` rejects immediately once `C` is saturated (no queue). `Q>0` lets up
  to `Q` requests busy-wait (in 0.005s steps) for a slot before rejecting.

The rejection rate and where it kicks in are the open model's most
important result: "at what arrival rate does the server start dropping
load?" Find it by sweeping `--arrival-rate` upward and watching the
rejected share.

## Same scenario, two models — chat-replay example

| | `chat-replay --model closed` | `chat-replay --model open` |
| --- | --- | --- |
| users | N always connected | sessions drawn from the pool, then leave |
| per-user lifetime | forever (dialogue wraps) | `--session-ops` ops per session |
| concurrency | N (fixed) | emergent (arrival vs service rate) |
| timing | think time only | Poisson arrival + think |
| rejections | none | `queue_full` rejections possible |
| models reality | "20 chat windows kept open" | "customers starting support chats at random" |

Both consume the *same* `plan()` — recall order, add content, and
`--search-every` cadence are identical. Only who fires when differs.

## When to use which

- **Closed** — measure peak throughput/latency at a fixed concurrency; the
  standard load-test shape. "How fast is the server when N users never stop?"
- **Open** — observe behavior under realistic arrival patterns and find the
  overload/rejection threshold. Sweep `--arrival-rate` up and watch
  `queue_full` rejections climb — the point where they start is the
  server's effective capacity.
- **Most realistic chatbot load** — `chat-replay --model open`: customers
  arrive per a Poisson process, each session recalls then answers/ingests a
  slice of the conversation, under the congestion policy.
