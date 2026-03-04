#![allow(dead_code, unused_imports)]

use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;
use shakmaty::Color as ChessColor;
use shakmaty::{Chess, File, Position as ChessPosition, Rank, Role, Square};

const LIGHT_SQ_BG: Color = Color::Rgb(240, 217, 181);
const DARK_SQ_BG: Color = Color::Rgb(181, 136, 99);
const CURSOR_BG: Color = Color::Yellow;
const SELECTED_BG: Color = Color::LightBlue;
const TARGET_BG: Color = Color::LightGreen;
const CELL_W: u16 = 4;
const CELL_H: u16 = 2;
pub const BOARD_WIDTH: u16 = 8 * CELL_W + 2; // 34 (incl. rank labels)
pub const BOARD_HEIGHT: u16 = 8 * CELL_H + 1; // 17 (incl. file labels)

pub struct BoardState {
    pub cursor: (u8, u8),
    pub selected: Option<Square>,
    pub legal_targets: Vec<Square>,
    pub flipped: bool,
}

impl Default for BoardState {
    fn default() -> Self {
        Self {
            cursor: (4, 1),
            selected: None,
            legal_targets: Vec::new(),
            flipped: false,
        }
    }
}

pub struct BoardWidget<'a> {
    pub position: &'a Chess,
    pub state: &'a BoardState,
}

impl Widget for BoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        render_board(self.position, self.state, area, buf);
    }
}

fn piece_glyph(role: Role, color: ChessColor) -> &'static str {
    match (color, role) {
        (ChessColor::White, Role::King) => "♔",
        (ChessColor::White, Role::Queen) => "♕",
        (ChessColor::White, Role::Rook) => "♖",
        (ChessColor::White, Role::Bishop) => "♗",
        (ChessColor::White, Role::Knight) => "♘",
        (ChessColor::White, Role::Pawn) => "♙",
        (ChessColor::Black, Role::King) => "♚",
        (ChessColor::Black, Role::Queen) => "♛",
        (ChessColor::Black, Role::Rook) => "♜",
        (ChessColor::Black, Role::Bishop) => "♝",
        (ChessColor::Black, Role::Knight) => "♞",
        (ChessColor::Black, Role::Pawn) => "♟",
    }
}

fn render_board(position: &Chess, state: &BoardState, area: Rect, buf: &mut Buffer) {
    // Rank labels (left edge)
    for dr in 0u16..8 {
        let label = if !state.flipped {
            (b'8' - dr as u8) as char
        } else {
            (b'1' + dr as u8) as char
        };
        let x = area.x;
        let y = area.y + dr * CELL_H;
        if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
            cell.set_symbol(&label.to_string());
        }
    }

    // File labels (bottom row)
    let label_row = area.y + 8 * CELL_H;
    for dc in 0u16..8 {
        let label = if !state.flipped {
            (b'a' + dc as u8) as char
        } else {
            (b'h' - dc as u8) as char
        };
        let x = area.x + 2 + dc * CELL_W + 1;
        if let Some(cell) = buf.cell_mut(Position::new(x, label_row)) {
            cell.set_symbol(&label.to_string());
        }
    }

    // Squares
    for rank in Rank::ALL {
        for file in File::ALL {
            let sq = Square::from_coords(file, rank);
            let file_idx = usize::from(file) as u8;
            let rank_idx = usize::from(rank) as u8;

            let (screen_file, screen_rank) = if !state.flipped {
                (file_idx as u16, 7 - rank_idx as u16)
            } else {
                (7 - file_idx as u16, rank_idx as u16)
            };

            let col = area.x + 2 + screen_file * CELL_W;
            let row = area.y + screen_rank * CELL_H;

            // Background priority: cursor > selected > legal target > square color
            let is_cursor = state.cursor == (file_idx, rank_idx);
            let is_selected = state.selected == Some(sq);
            let is_target = state.legal_targets.contains(&sq);
            let is_light = (file_idx + rank_idx) % 2 == 1;

            let bg = if is_cursor {
                CURSOR_BG
            } else if is_selected {
                SELECTED_BG
            } else if is_target {
                TARGET_BG
            } else if is_light {
                LIGHT_SQ_BG
            } else {
                DARK_SQ_BG
            };

            // Fill 4×2 cell background
            let cell_rect = Rect::new(col, row, CELL_W, CELL_H);
            let fill_rect = cell_rect.intersection(buf.area);
            if !fill_rect.is_empty() {
                buf.set_style(fill_rect, Style::new().bg(bg));
            }

            // Piece glyph at (col+1, row)
            if let Some(piece) = position.board().piece_at(sq) {
                let glyph = piece_glyph(piece.role, piece.color);
                let fg = match piece.color {
                    ChessColor::White => Color::Rgb(220, 220, 220),
                    ChessColor::Black => Color::Rgb(30, 10, 10),
                };
                if let Some(cell) = buf.cell_mut(Position::new(col + 1, row)) {
                    cell.set_symbol(glyph).set_fg(fg);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::widgets::Widget;
    use shakmaty::Chess;

    fn rendered_board() -> Buffer {
        let pos = Chess::default();
        let state = BoardState::default();
        let area = Rect::new(0, 0, BOARD_WIDTH, BOARD_HEIGHT);
        let mut buf = Buffer::empty(area);
        BoardWidget {
            position: &pos,
            state: &state,
        }
        .render(area, &mut buf);
        buf
    }

    #[test]
    fn board_widget_renders_without_panic() {
        let buf = rendered_board();
        // A1: file=0, rank=0 → col=2, row=14; (0+0)%2==0 → dark square
        assert_eq!(buf[(2, 14)].bg, DARK_SQ_BG);
    }

    #[test]
    fn board_widget_initial_rook_on_a1() {
        let buf = rendered_board();
        // White Rook glyph at (col+1, row) = (3, 14)
        assert_eq!(buf[(3, 14)].symbol(), "♖");
    }
}
