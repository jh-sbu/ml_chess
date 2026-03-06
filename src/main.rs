mod agents;
mod app;
mod chess;
mod eval;
mod training;
mod tui;

use std::path::PathBuf;
use std::time::Instant;

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
        /// Use MCTS agent instead of negamax
        #[arg(long)]
        mcts: bool,
        /// Minimum MCTS simulations per move (default 100)
        #[arg(long)]
        simulations: Option<u32>,
        /// Policy model checkpoint path (without extension)
        #[arg(long)]
        policy_model: Option<PathBuf>,
    },
    /// AI vs AI spectator mode
    Watch {
        #[arg(long)]
        depth: Option<u32>,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        mcts: bool,
        #[arg(long)]
        simulations: Option<u32>,
        #[arg(long)]
        policy_model: Option<PathBuf>,
    },
    /// Train neural network
    Train {
        #[arg(long)]
        games: Option<u32>,
        #[arg(long)]
        epochs: Option<u32>,
        /// Output path for saved value model (without extension)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Use MCTS self-play for training
        #[arg(long)]
        mcts: bool,
        /// MCTS simulations per move during self-play
        #[arg(long)]
        simulations: Option<u32>,
        /// Output path for saved policy model (without extension)
        #[arg(long)]
        policy_output: Option<PathBuf>,
        /// Suppress all training output
        #[arg(short, long)]
        quiet: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => run_menu(),
        Some(Commands::Play { depth, model, mcts, simulations, policy_model }) => {
            if mcts {
                run_play_mcts(simulations.unwrap_or(100), model, policy_model)
            } else {
                run_play(depth.unwrap_or(3), model)
            }
        }
        Some(Commands::Watch { depth, model, mcts, simulations, policy_model }) => {
            if mcts {
                run_watch_mcts(simulations.unwrap_or(100), model, policy_model)
            } else {
                run_watch(depth.unwrap_or(3), model)
            }
        }
        Some(Commands::Train { games, epochs, output, mcts, simulations, policy_output, quiet }) => {
            let output = output.unwrap_or_else(|| PathBuf::from("chess_model"));
            if mcts {
                let policy_out = policy_output.unwrap_or_else(|| PathBuf::from("chess_policy"));
                run_train_mcts(games.unwrap_or(100), epochs.unwrap_or(10), simulations.unwrap_or(50), output, policy_out, quiet)
            } else {
                run_train(games.unwrap_or(100), epochs.unwrap_or(10), output, quiet)
            }
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

fn run_play_mcts(
    simulations: u32,
    value_model: Option<PathBuf>,
    policy_model: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new_with_mcts(
        crate::tui::menu::MenuSelection::HumanVsAI,
        simulations,
        value_model,
        policy_model,
    )
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

fn run_watch_mcts(
    simulations: u32,
    value_model: Option<PathBuf>,
    policy_model: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = app::App::new_with_mcts(
        crate::tui::menu::MenuSelection::AIVsAI,
        simulations,
        value_model,
        policy_model,
    )
    .run(&mut terminal);
    ratatui::restore();
    result
}

fn run_train(games: u32, epochs: u32, output: PathBuf, quiet: bool) -> anyhow::Result<()> {
    if !quiet {
        eprintln!("Generating {games} self-play games...");
    }
    let sp_start = Instant::now();
    let (records, stats) = training::self_play::generate_games(games as usize, 1, quiet);
    let sp_elapsed = sp_start.elapsed();
    if !quiet {
        eprintln!(
            "Self-play complete in {:.0?} — {} wins / {} draws / {} losses, avg {:.0} moves/game",
            sp_elapsed, stats.wins, stats.draws, stats.losses, stats.avg_game_len,
        );
    }

    #[cfg(feature = "train")]
    {
        use training::trainer::{TrainingConfig, train};
        let config =
            TrainingConfig { epochs: epochs as usize, batch_size: 64, lr: 1e-3, policy_lr: None };
        if !quiet {
            eprintln!("Training for {epochs} epochs...");
        }
        let train_start = Instant::now();
        train(config, &records, &output, quiet)?;
        if !quiet {
            eprintln!("Training complete in {:.0?}", train_start.elapsed());
            eprintln!("Model saved to {}", output.display());
        }
    }
    #[cfg(not(feature = "train"))]
    {
        let _ = (epochs, output, records);
        eprintln!("Tip: run with `--features train` to enable the training loop.");
    }
    Ok(())
}

fn run_train_mcts(
    games: u32,
    epochs: u32,
    simulations: u32,
    output: PathBuf,
    policy_output: PathBuf,
    quiet: bool,
) -> anyhow::Result<()> {
    if !quiet {
        eprintln!("Generating {games} MCTS self-play games ({simulations} sims/move)...");
    }
    let sp_start = Instant::now();
    let (records, stats) =
        training::self_play::generate_mcts_games(games as usize, simulations, quiet);
    let sp_elapsed = sp_start.elapsed();
    if !quiet {
        eprintln!(
            "Self-play complete in {:.0?} — {} wins / {} draws / {} losses, avg {:.0} moves/game",
            sp_elapsed, stats.wins, stats.draws, stats.losses, stats.avg_game_len,
        );
    }

    #[cfg(feature = "train")]
    {
        use training::trainer::{TrainingConfig, train_policy};
        let config =
            TrainingConfig { epochs: epochs as usize, batch_size: 64, lr: 1e-3, policy_lr: None };
        if !quiet {
            eprintln!("Training value + policy nets for {epochs} epochs...");
        }
        let train_start = Instant::now();
        train_policy(&config, &records, &output, &policy_output, quiet)?;
        if !quiet {
            eprintln!("Training complete in {:.0?}", train_start.elapsed());
            eprintln!("Value model saved to {}", output.display());
            eprintln!("Policy model saved to {}", policy_output.display());
        }
    }
    #[cfg(not(feature = "train"))]
    {
        let _ = (epochs, output, policy_output, records);
        eprintln!("Tip: run with `--features train` to enable the training loop.");
    }
    Ok(())
}
