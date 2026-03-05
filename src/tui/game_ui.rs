#![allow(dead_code, unused_imports)]

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, List, ListItem, Paragraph, Widget};
use shakmaty::Color as ChessColor;
use shakmaty::{Chess, Position};

use crate::chess::board::GameState;
use crate::chess::moves::move_to_san;
use crate::tui::board_widget::{BOARD_WIDTH, BoardState, BoardWidget};

pub struct GameUiState<'a> {
    pub game_state: &'a GameState,
    pub board_state: &'a BoardState,
    pub white_name: &'a str,
    pub black_name: &'a str,
    pub status: &'a str,
}

pub fn render_game_ui(ui: &GameUiState<'_>, area: Rect, buf: &mut Buffer) {
    let [board_area, sidebar_area] =
        Layout::horizontal([Constraint::Length(BOARD_WIDTH), Constraint::Min(0)]).areas(area);

    BoardWidget {
        position: &ui.game_state.position,
        state: ui.board_state,
    }
    .render(board_area, buf);

    render_sidebar(ui, sidebar_area, buf);
}

fn render_sidebar(ui: &GameUiState<'_>, area: Rect, buf: &mut Buffer) {
    let block = Block::bordered();
    let inner = block.inner(area);
    block.render(area, buf);

    let [players_area, status_area, history_area] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .areas(inner);

    render_players(ui, players_area, buf);
    render_status(ui, status_area, buf);
    render_history(ui, history_area, buf);
}

fn render_players(ui: &GameUiState<'_>, area: Rect, buf: &mut Buffer) {
    let turn = ui.game_state.side_to_move();

    let black_style = if turn == ChessColor::Black {
        Style::new().add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    let white_style = if turn == ChessColor::White {
        Style::new().add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    let text = Text::from(vec![
        Line::from(Span::styled(format!("♟ {}", ui.black_name), black_style)),
        Line::from(""),
        Line::from(Span::styled(format!("♙ {}", ui.white_name), white_style)),
    ]);

    Paragraph::new(text)
        .block(Block::bordered().title("Players"))
        .render(area, buf);
}

fn render_status(ui: &GameUiState<'_>, area: Rect, buf: &mut Buffer) {
    let style = if ui.game_state.is_checkmate() {
        Style::new().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    Paragraph::new(ui.status)
        .style(style)
        .block(Block::bordered().title("Status"))
        .render(area, buf);
}

fn render_history(ui: &GameUiState<'_>, area: Rect, buf: &mut Buffer) {
    let mut pos = Chess::default();
    let mut items: Vec<ListItem> = Vec::new();
    let mut current_white = String::new();
    let mut move_num = 1u32;

    for (i, mv) in ui.game_state.history.iter().enumerate() {
        let san = move_to_san(&pos, mv);
        pos = match pos.play(mv) {
            Ok(new_pos) => new_pos,
            Err(_) => break,
        };

        if i % 2 == 0 {
            // White's move — start a new pair
            current_white = format!("{}. {}", move_num, san);
        } else {
            // Black's move — complete the pair
            items.push(ListItem::new(format!("{} {}", current_white, san)));
            current_white = String::new();
            move_num += 1;
        }
    }
    // Trailing white move with no reply yet
    if !current_white.is_empty() {
        items.push(ListItem::new(current_white));
    }

    List::new(items)
        .block(Block::bordered().title("Moves"))
        .render(area, buf);
}
