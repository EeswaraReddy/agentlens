# LinkedIn Post — AgentLens Launch

---

## POST (SHORT VERSION — best for engagement)

---

Most AI agents don't fail on the model.
They fail because nobody can see what the model actually did.

I built **AgentLens** to fix that.

It's an open-source observability + eval layer for AI agents.
Zero cloud. Zero vendor lock-in. One `pip install`.

What it does in 60 seconds:
→ traces every LLM call and tool call with timing + cost
→ catches behavior drift with declarative eval gates
→ plugs into Strands, LangGraph, OpenAI Agents SDK — one line each
→ exports to AWS X-Ray, Grafana Tempo, Jaeger via OpenTelemetry

Real output from a shopping assistant I traced today:

```
🤖 [agent] shopping-router        780ms
   🧠 [llm  ] classify_intent     82tok  $0.000004   332ms
   🔧 [tool ] get_user_history     
   🔧 [tool ] search_products     
   🔧 [tool ] check_stock         
   📍 [event] stock_checked       
   🔧 [tool ] validate_coupon     
   🧠 [llm  ] generate_recommendation  299tok  $0.000019   447ms

Total: 381 tokens · $0.000024 · 10/10 evals passed
```

The whole run cost less than 1/100th of a cent.
The evals told me the agent checked stock BEFORE recommending — every time.

If you're building agents and flying blind on what they do in production,
this is the layer you're missing.

MIT licensed. Free forever.

👇 GitHub + free cheat sheet in the comments.

#AI #AIAgents #MachineLearning #Python #OpenSource #LLM #AgentOps #MLOps #Observability

---

## POST (LONG VERSION — for newsletter / article)

---

**88% of AI agent projects never reach production.**

After building a few myself, I stopped believing it was a prompt problem or a model problem.

The demos worked. The prompts were fine.

What was missing was everything *around* the agent — the ability to see what it did, prove it behaved correctly, and stop it when it didn't.

So I built **AgentLens**.

---

**The problem it solves**

When your agent gives a bad answer at 2am, "the agent said something wrong" is useless. You need:

- Which LLM call produced the bad output?
- Did it actually check the database before answering?
- How many tokens did it burn? What did it cost?
- Is this a regression from yesterday's prompt change?

None of that is available in standard logging. AgentLens makes it all first-class.

---

**What it captures**

Every agent run becomes a span tree:

```
🤖 agent turn         (timing, metadata)
   🧠 LLM call        (tokens, cost, input prompt, output)
   🔧 tool call       (arguments, result, duration)
   🧠 LLM call        (tokens, cost, input, output)
   📍 event marker    (human approval, guardrail check)
```

Cost is computed automatically from token counts. A typical support agent run costs $0.000024. You see that per span and rolled up to the trace.

---

**The eval system — behavior as code**

This is the part I'm most proud of.

Instead of eyeballing outputs, you encode the behavior you require:

```python
suite = E.Suite("guardrails", [
    E.succeeded(),                          # run completed
    E.called_tool("lookup_order"),          # must check order
    E.never_called_tool("issue_refund"),    # must NOT refund without approval
    E.tool_before("lookup_order", "reply"), # correct sequence
    E.max_cost(0.01),                       # cost budget
    E.llm_judge("Reply is helpful and accurate", threshold=0.7),
])
assert suite.run(trace).passed  # fails CI if agent drifts
```

Drop that in your test suite. A prompt change that breaks behavior fails the build instead of surprising a customer.

---

**Framework adapters — one line each**

Strands, LangGraph, OpenAI Agents SDK all fire lifecycle events instead of letting you wrap calls. AgentLens has adapters for all three:

```python
# Strands — one line
agent = Agent(tools=[...], hooks=[make_agentlens_hook(tracer)])

# LangGraph — one line  
graph.invoke(input, config={"callbacks": [make_langgraph_handler(tracer)]})

# OpenAI Agents SDK — one line
add_trace_processor(make_agentlens_processor(tracer))
```

---

**Design comparison — find the cheapest architecture**

Before you commit to an agent design, run all your candidates through the same eval suite:

```
variant      spans  llm  tokens         cost    evals
minimal          3    1      68   $0.000003    5/5
rag              5    2     124   $0.000008    5/5
expensive        7    3     198   $0.000013    5/5

'expensive' costs 4.3x more than 'minimal' for the same eval result.
```

This one output has changed how I think about agent architecture.

---

**Zero cloud required**

The core has no dependencies. SQLite by default. Postgres when you need it.
Exports to AWS X-Ray, Grafana Tempo, Jaeger, Amazon Bedrock AgentCore via OpenTelemetry.

It's not competing with your existing observability stack — it feeds into it.

---

**Get started in 60 seconds**

```bash
pip install agentlens
python examples/shopping_assistant.py
agentlens ls
```

MIT licensed. Free forever. No account needed.

GitHub: [link in comments]
Free cheat sheet: [link in comments]

What observability problems are you hitting with your agents? Drop them below — I read every comment.

#AI #AIAgents #Python #OpenSource #LLM #AgentOps #MLOps #Observability #MachineLearning #LangChain #LangGraph

---

## COMMENT (pin after posting)

🔗 GitHub: https://github.com/your-username/agentlens
📄 Free cheat sheet (PDF): [your landing page URL]
🐍 Install: pip install agentlens
📖 Docs: python examples/user_guide.py

Questions? Drop them here — I answer everything.
