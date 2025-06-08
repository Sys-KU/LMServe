use std::fmt;

use chrono::Local;
use tracing_subscriber::{EnvFilter, fmt::format::Writer, fmt::time::FormatTime};

struct Date;

impl FormatTime for Date {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        write!(w, "{}", Local::now().format("%Y-%m-%dT%H:%M:%SZ"))
    }
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::Subscriber::builder()
        .with_timer(Date)
        .with_target(true)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();
}
