mod agents;
mod app;
mod chess;
mod eval;
mod tui;

fn main() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new().run(&mut terminal);
    ratatui::restore();
    result
}
