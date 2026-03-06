#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub use crate::agents::mcts::MoveVisit;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GameRecord {
    /// FEN string of each position visited during the game.
    pub positions: Vec<String>,
    /// Game outcome from White's perspective: +1.0 = White wins, 0.0 = draw, -1.0 = Black wins.
    pub outcome: f32,
}

/// Per-game record that also stores MCTS visit counts as policy training targets.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct MctsGameRecord {
    /// FEN string of each position visited (one per move made, excluding terminal).
    pub positions: Vec<String>,
    /// MCTS visit counts for each position; `move_visits[i]` corresponds to `positions[i]`.
    pub move_visits: Vec<Vec<MoveVisit>>,
    /// Game outcome from White's perspective: +1.0 / 0.0 / -1.0.
    pub outcome: f32,
}

pub fn save_records(records: &[GameRecord], path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer(file, records)?;
    Ok(())
}

pub fn load_records(path: &Path) -> anyhow::Result<Vec<GameRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let records = serde_json::from_reader(reader)?;
    Ok(records)
}

pub fn save_mcts_records(records: &[MctsGameRecord], path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer(file, records)?;
    Ok(())
}

pub fn load_mcts_records(path: &Path) -> anyhow::Result<Vec<MctsGameRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let records = serde_json::from_reader(reader)?;
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("ml_chess_dataset_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("records.json");

        let records = vec![
            GameRecord {
                positions: vec![
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".to_string(),
                ],
                outcome: 1.0,
            },
            GameRecord {
                positions: vec![
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                ],
                outcome: 0.0,
            },
        ];

        save_records(&records, &path).expect("save failed");
        let loaded = load_records(&path).expect("load failed");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].positions.len(), 2);
        assert_eq!(loaded[0].outcome, 1.0);
        assert_eq!(loaded[1].outcome, 0.0);

        fs::remove_file(path).ok();
    }

    #[test]
    fn save_and_load_mcts_records_roundtrip() {
        use crate::agents::mcts::MoveVisit;

        let dir = std::env::temp_dir().join("ml_chess_mcts_dataset_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcts_records.json");

        let records = vec![MctsGameRecord {
            positions: vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()],
            move_visits: vec![vec![
                MoveVisit { uci: "e2e4".to_string(), visits: 10 },
                MoveVisit { uci: "d2d4".to_string(), visits: 5 },
            ]],
            outcome: 1.0,
        }];

        save_mcts_records(&records, &path).expect("save failed");
        let loaded = load_mcts_records(&path).expect("load failed");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].positions.len(), 1);
        assert_eq!(loaded[0].move_visits[0].len(), 2);
        assert_eq!(loaded[0].move_visits[0][0].visits, 10);
        assert_eq!(loaded[0].outcome, 1.0);

        fs::remove_file(path).ok();
    }
}
