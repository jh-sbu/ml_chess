#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GameRecord {
    /// FEN string of each position visited during the game.
    pub positions: Vec<String>,
    /// Game outcome from White's perspective: +1.0 = White wins, 0.0 = draw, -1.0 = Black wins.
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
}
