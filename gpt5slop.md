

2084

Tool Use Without the Training Wheels: Why I Stopped Treating LLMs Like API Routers

If you watch most demos of “tool calling,” you’ll see the same choreography: the model picks a function, emits tidy JSON, gets a tidy answer, and moves on. It’s cute. It’s also the wrong shape for real work.

I didn’t learn that in a paper. I learned it the first time I asked an agent to write a sales deck, then asked it to add a chart, then a caption, then an appendix with the methodology, then to go find three more sources and re‑plot the chart with confidence bands. JSON was nowhere to be seen by the end. What I needed was a way for the model to hold onto a living object—a document, a presentation, a database connection—and keep working it like a chef returning to a sauce.

This is the story of how I built that. And why I stopped treating LLMs like function dispatchers and started treating them like interns who can pick up a client and keep it on the counter.

### The moment JSON stopped being enough

I used the classic function-calling pattern for a while. It’s great for one-off actions: “fetch the weather,” “lookup this SKU,” “translate this sentence.” But the minute the conversation becomes a project, the JSON turns into a straitjacket. You start writing prompts that basically teach the model to write a programming language inside a single call. If the task wants loops, conditionals, retries, backoff, or anything resembling a workflow, you either hand-wave or you reinvent a brittle mini-runtime in the prompt.

The second pattern—code execution—was a breath of fresh air. Let the model write Python, execute it, feed the output back. I loved it, and I still do. But code execution as a single-shot block has the same pathology: if you want a long artifact, you ask the model to compose a symphony in one breath. That’s not how humans write reports or presentations.

The fix was to stop being precious about “tools as pure functions” and let the agent own a client object across steps. A `Document`. A `Presentation`. A `sqlengine` bound to a particular database. State that persists, method calls that accrete, and room to change your mind.

### The harness I ended up building

I call it Maximum Agents. It’s a small layer around smolagents’ code executor that adds three things I couldn’t live without:

1) Steps you can stream and skim.
Every step is normalized into parts—thinking, code, outputs, tool calls—so the UI (or a database) can follow along without parsing spaghetti. You see where the model thought, where it executed, and what came back.

```323:372:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximum_agents/src/maximum_agents/base.py
def format_step(self,step_number : int, step: ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput | ActionStep | PlanningStep | FinalAnswerStep) -> StepT | ResultT[T]:
    ...
    if isinstance(step, ActionStep):
        parts = []
        if step.is_final_answer and step.action_output is not None:
            return ResultT[T](answer=self.final_answer_model.model_validate(step.action_output, context=self.final_answer_context))
        ...
        return StepT(step_number=step_number, parts=deduplicate_parts(parts))
```

2) Final answers with a backbone.
The last thing an agent does is call a Pydantic-shaped “final answer” tool. The result is not vibes; it’s a schema. When your agent says it produced two documents and why, you can trust the structure.

```225:242:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximum_agents/src/maximum_agents/pydantic_final_answer_tools.py
class PydanticFinalAnswerTool(FinalAnswerTool):
    def forward(self, answer: dict[str, Any]) -> dict[str, Any]:
        data = self.model_pydantic.model_validate(answer, context=self.context)
        return data.model_dump()
```

3) A builder that treats “place” as a first-class concept.
Work happens in a workspace—a directory the agent owns for the session—so when it saves `report.docx` or `chart.png`, the frontend knows exactly where to find it and the backend knows exactly how to serve it. While I’m at it, I can inject a database and give the agent a proper SQL tool bound to that database.

```151:167:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximum_agents/src/maximum_agents/builders/builder.py
def add_database(self, datastore: MaximumDataStore, database_id: str) -> 'AgentBuilder':
    self._datastore = datastore
    self._database_id = database_id
    sql_tool = DatabaseTool(database_id, datastore)
    self.additional_tools.append(sql_tool)
```

There’s a little assist in the document types too: if the agent returns a relative path, the framework resolves it to an absolute path using a context-aware finder, so your UI doesn’t have to guess.

```14:21:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximum_agents/src/maximum_agents/document_types.py
class DocumentT(BaseModel):
    def model_post_init(self, __context: Optional[dict[str, Any]]) -> None:
        if self.absolute_path is None:
            if isinstance(__context, dict) and 'document_finder' in __context:
                document_finder = __context['document_finder']
                self.absolute_path = document_finder(self.path)
```

### The app I actually ship

Frameworks are nice; apps are nicer. MaximumResearch is the full thing: a FastAPI backend that streams steps via SSE and persists to Postgres, and a frontend that renders the play-by-play and opens the document the agent just wrote—right in the browser.

The backend kicks off an analysis, binds the agent to a workspace directory, and starts streaming. No drama.

```231:237:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximumresearch/backend/src/main.py
result = await run_agent_analysis(
    task=task,
    output_dir=analysis_output_dir,
    model=analysis.model,
    on_step=log_step
)
```

The agent itself is opinionated in a useful way. It has a Word client, a PowerPoint client, and a web search tool. And it’s told—politely but firmly—to save its work and tell us where it put it.

```395:404:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximumresearch/backend/src/research_agent.py
agent = builder.build_agent(
    model=model,
    system_prompt=system_prompt,
    final_answer_model=DocumentsT,
    final_answer_description="Generated documents with sales analysis and recommendations",
    additional_authorized_imports=[...],
    tools=tools
)
```

On the frontend, I avoided the “download and pray” experience. If the agent writes a `.docx`, I try to view it in-browser. SuperDoc made that possible. It’s not perfect, but it’s good enough to feel like a living document, right there in the conversation.

```22:28:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximumresearch/frontend/src/components/SuperDocViewer.tsx
export function SuperDocViewer({ document, analysisId }: SuperDocViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const superDocRef = useRef<any>(null)
  ...
}
```

And yes, the agent can talk SQL. The datastore is a local DuckDB backend, wrapped in a tool called `sqlengine`. The model can ask the database questions like a person would—and then stick the answers into a paragraph, a table, or a chart.

```32:39:/Users/lukas/Desktop/Fuckaroundy/researchagent/maximum_agents/src/maximum_agents/datastore/core.py
def sql_engine(
    self, 
    database_id: str, 
    sql_query: str, 
    optional_params: Optional[Dict[str, Any]] = None,
    access_control: Optional[AccessControlT] = None
) -> pd.DataFrame:
```

### What it feels like to use

A normal flow: “Analyze last quarter’s pipeline. Identify stalls by segment. Draft a two-page narrative for the sales leadership call. Include one chart and a list of remediation ideas.”

The agent searches, reads, writes a couple of paragraphs. It realizes it wants a chart, grabs a DataFrame from SQL, draws the thing, saves it to disk, and slots it into the doc. It revisits the intro because the chart changed the story. It adds a sources section because the system prompt nags it to. It saves, returns a `DocumentsT` that tells me what the file is and why I should open it.

There’s no “call X with Y” performance. There’s also no 300‑line code block that births a perfect doc in one go. It keeps a `Document` on the counter and returns to it like a person would.

### The boring decisions that pay off

- Hooks everywhere. I can change the system prompt—add the task, add the final-answer schema, sneak in a database description—without touching agent code.
- Authorized imports. The agent can’t import the world; it imports exactly what it needs: `docx`, `pptx`, `matplotlib`, `pandas`, `numpy`, `seaborn`, web search, and friends.
- Typed finals. The last mile is always structured. The UI isn’t guessing whether we got a doc, a deck, or an apology; it’s reading a schema.
- Workspaces as reality. The output directory is real. The file paths are real. The download link works.

### Where this sits in the bigger conversation

There’s a place for JSON function calling (it’s lovely when you really do just want a single move). There’s a place for ReAct-style loops when you want the model to think in public and interleave actions with observations. And there’s a place for graph orchestrators when your workflow has multiple independent nodes.

But when the output is an artifact and the work is iterative, the most boring answer wins: hold onto a client, keep calling methods, and save often. That’s what Maximum Agents gives you: an agent that can write, revise, and ship like a person who learned to save their work after losing a draft once.

### If you want to poke around

- The agent harness, hooks, datastore, and builder live in `maximum_agents`.
- The full app (FastAPI backend with SSE + React frontend + SuperDoc viewer) lives in `maximumresearch`.
- If you open the code, you’ll see very few magic tricks, a lot of explicitness, and a bias toward structures that survive long conversations.

I don’t think tool use should feel like filing a form. It should feel like making something. The minute I let the agent keep a handle on a living object and come back to it, the whole thing stopped feeling like “AI calling APIs” and started feeling like work getting done.

If you want me to drop a quickstart here (end‑to‑end command and a sample dataset), say the word and I’ll add it.
