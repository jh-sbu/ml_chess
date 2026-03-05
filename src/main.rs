mod agents;
mod app;
mod chess;
mod eval;
mod tui;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ml_chess", about = "Terminal chess with ML agents")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Human vs AI — TUI game
    Play {
        #[arg(long)]
        depth: Option<u32>,
    },
    /// AI vs AI spectator mode
    Watch {
        #[arg(long)]
        depth: Option<u32>,
    },
    /// Train neural network (Phase 11)
    Train {
        #[arg(long)]
        games: Option<u32>,
        #[arg(long)]
        epochs: Option<u32>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => run_menu(),
        Some(Commands::Play { depth }) => run_play(depth.unwrap_or(3)),
        Some(Commands::Watch { depth }) => run_watch(depth.unwrap_or(3)),
        Some(Commands::Train { games, epochs }) => {
            run_train(games.unwrap_or(100), epochs.unwrap_or(10))
        }
    }
}

fn run_menu() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new().run(&mut terminal);
    ratatui::restore();
    result
}

fn run_play(depth: u32) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result =
        app::App::new_with_mode(crate::tui::menu::MenuSelection::HumanVsAI, depth)
            .run(&mut terminal);
    ratatui::restore();
    result
}

fn run_watch(depth: u32) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result =
        app::App::new_with_mode(crate::tui::menu::MenuSelection::AIVsAI, depth)
            .run(&mut terminal);
    ratatui::restore();
    result
}

fn run_train(games: u32, epochs: u32) -> anyhow::Result<()> {
    eprintln!("Training not yet implemented (Phase 11). games={games}, epochs={epochs}");
    Ok(())
}
