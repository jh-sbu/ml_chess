mod agents;
mod app;
mod chess;
mod eval;
mod training;
mod tui;

use std::path::PathBuf;

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
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// AI vs AI spectator mode
    Watch {
        #[arg(long)]
        depth: Option<u32>,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Train neural network
    Train {
        #[arg(long)]
        games: Option<u32>,
        #[arg(long)]
        epochs: Option<u32>,
        /// Output path for saved model (without extension)
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => run_menu(),
        Some(Commands::Play { depth, model }) => run_play(depth.unwrap_or(3), model),
        Some(Commands::Watch { depth, model }) => run_watch(depth.unwrap_or(3), model),
        Some(Commands::Train { games, epochs, output }) => {
            let output = output.unwrap_or_else(|| PathBuf::from("chess_model"));
            run_train(games.unwrap_or(100), epochs.unwrap_or(10), output)
        }
    }
}

fn run_menu() -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new().run(&mut terminal);
    ratatui::restore();
    result
}

fn run_play(depth: u32, model: Option<PathBuf>) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new_with_model(crate::tui::menu::MenuSelection::HumanVsAI, depth, model)
        .run(&mut terminal);
    ratatui::restore();
    result
}

fn run_watch(depth: u32, model: Option<PathBuf>) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new_with_model(crate::tui::menu::MenuSelection::AIVsAI, depth, model)
        .run(&mut terminal);
    ratatui::restore();
    result
}

fn run_train(games: u32, epochs: u32, output: PathBuf) -> anyhow::Result<()> {
    eprintln!("Generating {games} self-play games...");
    let records = training::self_play::generate_games(games as usize, 1);
    eprintln!(
        "Generated {} games with {} total positions.",
        records.len(),
        records.iter().map(|r| r.positions.len()).sum::<usize>()
    );

    #[cfg(feature = "train")]
    {
        use training::trainer::{TrainingConfig, train};
        let config = TrainingConfig { epochs: epochs as usize, batch_size: 64, lr: 1e-3 };
        eprintln!("Training for {epochs} epochs...");
        train(config, &records, &output)?;
        eprintln!("Model saved to {}", output.display());
    }
    #[cfg(not(feature = "train"))]
    {
        let _ = (epochs, output);
        eprintln!("Tip: run with `--features train` to enable the training loop.");
    }
    Ok(())
}
