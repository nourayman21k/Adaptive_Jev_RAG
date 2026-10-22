import pandas as pd
import streamlit as st

from common import load_result

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
                 "context_precision": "Context precision", "context_recall": "Context recall"}

st.caption("Measured offline in the project notebooks. Every number comes from the files in `results/`.")

rag_tab, router_tab, jev_tab = st.tabs(
    ["RAG quality", "JEV vs BERT router", "Making JEV cheaper"],
    on_change="rerun",
)

if rag_tab.open:
    with rag_tab:
        summary = load_result("ragas_summary.json")
        scores = load_result("ragas_scores.csv")
        if summary is None or scores is None:
            st.info("No RAGAS results found. Run the evaluation section of adaptive.ipynb.", icon=":material/info:")
        else:
            s = summary["summary"]
            st.markdown(f"RAGAS on **{s['n_questions']} questions**, one per paper. Generator `{s['generator']}`, "
                        f"judged by `{s['judge']}`.")
            for col, m in zip(st.columns(4), METRICS):
                col.metric(METRIC_LABELS[m], f"{s['scores_mean'][m]:.3f}", border=True,
                           chart_data=scores[m].fillna(0).tolist(), chart_type="bar",
                           help=f"Mean over the questions; lowest {s['scores_min'][m]:.3f}")
            st.dataframe(
                scores[["paper", *METRICS]],
                hide_index=True,
                column_config={
                    "paper": st.column_config.TextColumn("Paper", width="medium"),
                    **{m: st.column_config.ProgressColumn(METRIC_LABELS[m], format="%.2f", min_value=0, max_value=1)
                       for m in METRICS},
                },
                alt="RAGAS scores per question",
            )
            p = s["pipeline"]
            lat_col, tok_col, ctx_col, web_col = st.columns(4)
            lat_col.metric("Mean latency", f"{p['mean_latency_s']:.1f} s", border=True,
                           help="Mostly Groq free-tier rate-limit waits; 7.2 s unthrottled")
            tok_col.metric("Tokens per question", f"{p['mean_total_tokens']:,.0f}", border=True)
            ctx_col.metric("Chunks kept by the grader", f"{p['mean_contexts_after_grading']:.1f} of 6", border=True)
            web_col.metric("Web fallbacks", f"{p['web_fallbacks']} of {s['n_questions']}", border=True)

if router_tab.open:
    with router_tab:
        jev = load_result("router_jev.json")
        bert = load_result("router_bert.json")
        if jev is None or bert is None:
            st.info("No router results found. Run the evaluation sections of both notebooks.", icon=":material/info:")
        else:
            j, b = jev["summary"], bert["summary"]
            st.markdown(f"Both routers on the same **{j['n_test']} held-out queries**, one query at a time.")
            acc_col, lat_col, tok_col, mem_col = st.columns(4)
            acc_col.metric("Accuracy (BERT)", f"{b['accuracy']:.1%}", f"{b['accuracy'] - j['accuracy']:+.1%} vs JEV",
                           border=True)
            lat_col.metric("Latency (BERT)", f"{b['latency']['mean_ms']:.1f} ms",
                           f"{j['latency']['mean_ms'] / b['latency']['mean_ms']:.1f}× faster", border=True)
            tok_col.metric("Tokens per query (BERT)", f"{b['tokens']['mean_input_per_query']:.0f}",
                           f"{j['tokens']['mean_input_per_query'] / b['tokens']['mean_input_per_query']:.0f}× fewer",
                           border=True)
            mem_col.metric("Peak GPU memory (BERT)", f"{b['peak_gpu_memory_mb'] / 1024:.1f} GB",
                           f"{j['peak_gpu_memory_mb'] / b['peak_gpu_memory_mb']:.1f}× less", border=True)
            table = pd.DataFrame({
                "JEV (Qwen3.5-4B, zero-shot)": {
                    "Accuracy": f"{j['accuracy']:.1%}", "Macro F1": f"{j['macro_f1']:.3f}",
                    "Mean latency": f"{j['latency']['mean_ms']:.1f} ms", "p95 latency": f"{j['latency']['p95_ms']:.1f} ms",
                    "Input tokens / query": f"{j['tokens']['mean_input_per_query']:.1f}",
                    "Parameters": f"{j['parameters'] / 1e6:,.0f} M", "Peak GPU memory": f"{j['peak_gpu_memory_mb']:,.0f} MB",
                    "Labeled training data": "none",
                },
                "BERT (fine-tuned)": {
                    "Accuracy": f"{b['accuracy']:.1%}", "Macro F1": f"{b['macro_f1']:.3f}",
                    "Mean latency": f"{b['latency']['mean_ms']:.1f} ms", "p95 latency": f"{b['latency']['p95_ms']:.1f} ms",
                    "Input tokens / query": f"{b['tokens']['mean_input_per_query']:.1f}",
                    "Parameters": f"{b['parameters'] / 1e6:,.0f} M", "Peak GPU memory": f"{b['peak_gpu_memory_mb']:,.0f} MB",
                    "Labeled training data": f"{b['training_examples']} queries ({b['train_s']:.1f} s training)",
                },
            })
            st.table(table)
            latency = pd.DataFrame({"router": ["JEV", "BERT"],
                                    "mean latency (ms)": [j["latency"]["mean_ms"], b["latency"]["mean_ms"]]})
            st.bar_chart(latency, x="router", y="mean latency (ms)", horizontal=True, height=180)
            misses = [r for r in bert["rows"] if r["predicted"] != r["label"]]
            if misses:
                st.caption("BERT's misroutes: " + "; ".join(
                    f"“{r['query']}” → {r['predicted']} ({r['confidence']:.0%})" for r in misses))

if jev_tab.open:
    with jev_tab:
        variants = load_result("router_jev_variants.json")
        if variants is None:
            st.info("No results found. Run jev_optimizations.ipynb.", icon=":material/info:")
        else:
            st.markdown("Qwen3.5-4B in bf16 and 4-bit, each run with the full prompt, a cached prompt prefix, "
                        "and all 30 queries in one batch. Every setup routed 30 of 30 correctly.")
            df = pd.DataFrame([{
                "setup": f"{v['model']} · {v['mode']}",
                "accuracy": v["accuracy"],
                "mean_ms": v["mean_ms"],
                "computed_tokens": v["computed_tokens_per_query"],
                "peak_gb": v["peak_gpu_mb"] / 1024,
            } for v in variants if v.get("mode") != "failed"])
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "setup": st.column_config.TextColumn("Setup", width="large"),
                    "accuracy": st.column_config.ProgressColumn("Accuracy", format="percent", min_value=0, max_value=1),
                    "mean_ms": st.column_config.NumberColumn("Mean latency", format="%.1f ms"),
                    "computed_tokens": st.column_config.NumberColumn("Tokens computed / query", format="%.0f"),
                    "peak_gb": st.column_config.NumberColumn("Peak GPU memory", format="%.1f GB"),
                },
                alt="Accuracy, latency, tokens and memory for each JEV setup",
            )
            st.bar_chart(df, x="setup", y="mean_ms", horizontal=True, x_label="mean latency (ms)", y_label="",
                         height=260)
            st.caption("4-bit cuts peak GPU memory by 60 % at about 20 % more latency; batching is the only large "
                       "latency win; prefix caching skips 77 % of the prompt tokens but barely changes latency.")
