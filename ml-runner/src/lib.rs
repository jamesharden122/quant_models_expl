#[cfg(feature = "server")]
pub mod backtest;
#[cfg(feature = "server")]
pub mod inference;
pub mod pipelines;
pub mod pyexec;
#[cfg(feature = "server")]
pub mod streaming;
pub mod surr_queries;
#[cfg(feature = "server")]
pub mod training;
use dioxus::prelude::*;

#[component]
fn TrainingForm() -> Element {
    let mut trainer_py = use_signal(|| "../ml-project/py/mls_lstm_trainer.py".to_string());
    let mut tfrecord_path = use_signal(|| "../tmp_data/data.tfrecord".to_string());
    let mut callable = use_signal(|| "train".to_string());

    let run_training = move |evt: FormEvent| {
        evt.prevent_default();
        // In web builds, you can wire this to server functions via fullstack routing.
        // Left as a stub to keep compilation simple.
        println!(
            "Submit clicked: trainer={}, tfrec={}, call={}",
            trainer_py(),
            tfrecord_path(),
            callable()
        );
    };

    rsx! {
        form { onsubmit: run_training,
            label { "Trainer Python file" }
            input { value: trainer_py(), oninput: move |evt| trainer_py.set(evt.value()) }
            label { "TFRecord file" }
            input { value: tfrecord_path(), oninput: move |evt| tfrecord_path.set(evt.value()) }
            label { "Callable name" }
            input { value: callable(), oninput: move |evt| callable.set(evt.value()) }
            button { r#type: "submit", "Run Training" }
        }
    }
}

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    TrainingForm,
}

#[component]
pub fn app() -> Element {
    rsx! { Router::<Route> {} }
}
