use dioxus::prelude::*;
use anyhow::Result;
use serde::Deserialize;
use surrealdb::{engine::any::Any, Surreal};
use ml_backend::{make_df_from_surreal, train_via_pyo3};
use polars::prelude::DataFrame;

#[derive(Deserialize)]
struct ModelRow {
    name: String,
}

async fn fetch_models(term: &str) -> Result<Vec<String>> {
    let db = Surreal::new::<Any>("http://localhost:8000").await?;
    db.use_ns("research").use_db("deeplearning").await?;
    let sql = format!("SELECT name FROM models WHERE name CONTAINS '{}';", term);
    let mut resp = db.query(sql).await?;
    let rows: Vec<ModelRow> = resp.take(0)?;
    Ok(rows.into_iter().map(|r| r.name).collect())
}

async fn train_model(model: String) -> Result<()> {
    let db = Surreal::new::<Any>("http://localhost:8000").await?;
    db.use_ns("research").use_db("deeplearning").await?;
    let sql = format!("SELECT * FROM data WHERE model = '{model}';");
    let df: DataFrame = make_df_from_surreal(&db, &sql).await?;
    let cfg = format!("{{\"target\":\"close\",\"epochs\":10,\"batch\":64,\"outdir\":\"models\"}}",
    );
    let _ = train_via_pyo3(&df, "trainer", "train_from_polars", &cfg)?;
    Ok(())
}

pub fn app(cx: Scope) -> Element {
    let search = use_state(cx, || String::new());
    let selected = use_state(cx, || None as Option<String>);
    let models = use_future(cx, search, |search| {
        let term = search.current().clone();
        async move { fetch_models(&term).await.unwrap_or_default() }
    });

    cx.render(rsx! {
        div { class: "model-trainer",
            input {
                value: "{search}",
                placeholder: "Search models...",
                oninput: move |e| search.set(e.value.clone()),
            }
            ul {
                models.value().unwrap_or(&vec![]).iter().map(|name| {
                    let name_clone = name.clone();
                    rsx! {
                        li {
                            onclick: move |_| selected.set(Some(name_clone.clone())),
                            "{name}"
                        }
                    }
                })
            }
            button {
                onclick: move |_| {
                    if let Some(name) = selected.get().clone() {
                        tokio::spawn(async move {
                            if let Err(e) = train_model(name).await {
                                eprintln!("training failed: {e}");
                            }
                        });
                    }
                },
                "Train"
            }
        }
    })
}
