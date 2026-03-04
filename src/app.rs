use std::sync::mpsc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Clear, Paragraph, Widget};
use ratatui::{DefaultTerminal, Frame};
use shakmaty::Color as ChessColor;
use shakmaty::{File, Outcome, Position, Rank, Role, Square};

use crate::agents::{Agent, MinimaxAgent, NegamaxAgent};
use crate::chess::board::GameState;
use crate::chess::Move;
use crate::tui::board_widget::BoardState;
use crate::tui::game_ui::{render_game_ui, GameUiState};
use crate::tui::menu::{MainMenu, MenuSelection};

pub enum Screen {
    Menu,
    Game,
    GameOver { message: String },
}

pub struct App {
    screen: Screen,
    menu: MainMenu,
    game: GameState,
    board_state: BoardState,
    white_name: String,
    black_name: String,
    ai_white: Option<Box<dyn Agent + Send>>,
    ai_black: Option<Box<dyn Agent + Send>>,
    ai_agent_rx: Option<mpsc::Receiver<(Move, Box<dyn Agent + Send>)>>,
    ai_thinking_color: Option<ChessColor>,
    ai_is_thinking: bool,
    status: String,
}

impl App {
    pub fn new() -> Self {
        Self {
            screen: Screen::Menu,
            menu: MainMenu::default(),
            game: GameState::new(),
            board_state: BoardState::default(),
            white_name: String::new(),
            black_name: String::new(),
            ai_white: None,
            ai_black: None,
            ai_agent_rx: None,
            ai_thinking_color: None,
            ai_is_thinking: false,
            status: String::new(),
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        loop {
            if event::poll(Duration::from_millis(50))? {
                let ev = event::read()?;
                if self.handle_event(ev) {
                    break;
                }
            }
            self.ai_tick();
            terminal.draw(|f| self.render(f))?;
        }
        Ok(())
    }

    fn handle_event(&mut self, ev: Event) -> bool {
        let Event::Key(KeyEvent { code, kind: KeyEventKind::Press, .. }) = ev else {
            return false;
        };
        match &self.screen {
            Screen::Menu => self.handle_key_menu(code),
            Screen::Game | Screen::GameOver { .. } => self.handle_key_game(code),
        }
    }

    fn handle_key_menu(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Up => {
                self.menu.move_up();
                false
            }
            KeyCode::Down => {
                self.menu.move_down();
                false
            }
            KeyCode::Enter => {
                let sel = self.menu.current();
                self.start_game(sel)
            }
            KeyCode::Char('q') | KeyCode::Esc => true,
            _ => false,
        }
    }

    fn handle_key_game(&mut self, key: KeyCode) -> bool {
        if let Screen::GameOver { .. } = &self.screen {
            self.screen = Screen::Menu;
            return false;
        }
        match key {
            KeyCode::Up => {
                let (f, r) = self.board_state.cursor;
                if !self.board_state.flipped {
                    if r < 7 {
                        self.board_state.cursor = (f, r + 1);
                    }
                } else if r > 0 {
                    self.board_state.cursor = (f, r - 1);
                }
                false
            }
            KeyCode::Down => {
                let (f, r) = self.board_state.cursor;
                if !self.board_state.flipped {
                    if r > 0 {
                        self.board_state.cursor = (f, r - 1);
                    }
                } else if r < 7 {
                    self.board_state.cursor = (f, r + 1);
                }
                false
            }
            KeyCode::Left => {
                let (f, r) = self.board_state.cursor;
                if !self.board_state.flipped {
                    if f > 0 {
                        self.board_state.cursor = (f - 1, r);
                    }
                } else if f < 7 {
                    self.board_state.cursor = (f + 1, r);
                }
                false
            }
            KeyCode::Right => {
                let (f, r) = self.board_state.cursor;
                if !self.board_state.flipped {
                    if f < 7 {
                        self.board_state.cursor = (f + 1, r);
                    }
                } else if f > 0 {
                    self.board_state.cursor = (f - 1, r);
                }
                false
            }
            KeyCode::Enter => {
                self.select_or_move();
                false
            }
            KeyCode::Char('f') => {
                self.board_state.flipped = !self.board_state.flipped;
                false
            }
            KeyCode::Esc => {
                self.board_state.selected = None;
                self.board_state.legal_targets.clear();
                false
            }
            KeyCode::Char('q') => {
                self.screen = Screen::Menu;
                self.ai_agent_rx = None;
                self.ai_is_thinking = false;
                false
            }
            _ => false,
        }
    }

    /// Returns true if the app should quit.
    fn start_game(&mut self, selection: MenuSelection) -> bool {
        match selection {
            MenuSelection::Train => {
                self.screen = Screen::GameOver {
                    message: "Training not implemented yet".to_string(),
                };
                return false;
            }
            MenuSelection::Quit => return true,
            _ => {}
        }

        self.game = GameState::new();
        self.board_state = BoardState::default();
        self.ai_agent_rx = None;
        self.ai_is_thinking = false;
        self.ai_thinking_color = None;

        match selection {
            MenuSelection::HumanVsHuman => {
                self.ai_white = None;
                self.ai_black = None;
                self.white_name = "White".to_string();
                self.black_name = "Black".to_string();
            }
            MenuSelection::HumanVsAI => {
                self.ai_white = None;
                self.ai_black = Some(Box::new(NegamaxAgent::new(3)));
                self.white_name = "Human".to_string();
                self.black_name = "AI (Negamax d3)".to_string();
            }
            MenuSelection::AIVsAI => {
                self.ai_white = Some(Box::new(NegamaxAgent::new(3)));
                self.ai_black = Some(Box::new(MinimaxAgent::new(3)));
                self.white_name = "AI (Negamax d3)".to_string();
                self.black_name = "AI (Minimax d3)".to_string();
            }
            _ => unreachable!(),
        }

        self.screen = Screen::Game;
        self.update_status();

        if self.ai_white.is_some() {
            self.start_ai_thinking();
        }

        false
    }

    fn cursor_to_square(&self) -> Square {
        let (f, r) = self.board_state.cursor;
        Square::from_coords(File::new(f as u32), Rank::new(r as u32))
    }

    fn is_human_turn(&self) -> bool {
        if self.ai_is_thinking {
            return false;
        }
        match self.game.side_to_move() {
            ChessColor::White => self.ai_white.is_none(),
            ChessColor::Black => self.ai_black.is_none(),
        }
    }

    fn select_or_move(&mut self) {
        if !self.is_human_turn() {
            return;
        }
        let to = self.cursor_to_square();

        if let Some(from) = self.board_state.selected {
            let legal = self.game.legal_moves();
            // Prefer queen promotion when multiple promotions exist
            let found = legal
                .iter()
                .filter(|mv| mv.from() == Some(from) && mv.to() == to)
                .max_by_key(|mv| if mv.promotion() == Some(Role::Queen) { 1 } else { 0 })
                .cloned();

            if let Some(mv) = found {
                self.board_state.selected = None;
                self.board_state.legal_targets.clear();
                self.apply_player_move(mv);
            } else {
                // Re-select if cursor is on own piece, otherwise clear
                let own_piece = self
                    .game
                    .position
                    .board()
                    .piece_at(to)
                    .is_some_and(|p| p.color == self.game.side_to_move());
                if own_piece {
                    self.do_select(to);
                } else {
                    self.board_state.selected = None;
                    self.board_state.legal_targets.clear();
                }
            }
        } else {
            let own_piece = self
                .game
                .position
                .board()
                .piece_at(to)
                .is_some_and(|p| p.color == self.game.side_to_move());
            if own_piece {
                self.do_select(to);
            }
        }
    }

    fn do_select(&mut self, sq: Square) {
        self.board_state.selected = Some(sq);
        let legal = self.game.legal_moves();
        self.board_state.legal_targets = legal
            .iter()
            .filter(|mv| mv.from() == Some(sq))
            .map(|mv| mv.to())
            .collect();
    }

    fn apply_player_move(&mut self, mv: Move) {
        if let Err(e) = self.game.apply_move(mv) {
            eprintln!("Error applying move: {e}");
            return;
        }
        self.board_state.selected = None;
        self.board_state.legal_targets.clear();

        if let Some(outcome) = self.game.outcome() {
            let msg = match outcome {
                Outcome::Decisive { winner } => format!(
                    "{} wins!",
                    match winner {
                        ChessColor::White => "White",
                        ChessColor::Black => "Black",
                    }
                ),
                Outcome::Draw => "Draw!".to_string(),
            };
            self.screen = Screen::GameOver { message: msg };
            return;
        }

        if self.game.halfmove_clock() >= 100 {
            self.screen = Screen::GameOver {
                message: "Draw (50-move rule)".to_string(),
            };
            return;
        }

        self.update_status();

        if matches!(self.screen, Screen::Game) {
            let next_is_ai = match self.game.side_to_move() {
                ChessColor::White => self.ai_white.is_some(),
                ChessColor::Black => self.ai_black.is_some(),
            };
            if next_is_ai {
                self.start_ai_thinking();
            }
        }
    }

    fn start_ai_thinking(&mut self) {
        let color = self.game.side_to_move();
        let agent = match color {
            ChessColor::White => self.ai_white.take(),
            ChessColor::Black => self.ai_black.take(),
        };
        let Some(mut agent) = agent else { return };

        let game_clone = self.game.clone();
        let (tx, rx) = mpsc::channel::<(Move, Box<dyn Agent + Send>)>();
        self.ai_agent_rx = Some(rx);
        self.ai_thinking_color = Some(color);
        self.ai_is_thinking = true;
        self.status = "AI is thinking...".to_string();

        std::thread::spawn(move || {
            let mv = agent.select_move(&game_clone, None);
            let _ = tx.send((mv, agent));
        });
    }

    fn ai_tick(&mut self) {
        let result = self.ai_agent_rx.as_ref().and_then(|rx| rx.try_recv().ok());

        if let Some((mv, agent)) = result {
            self.ai_agent_rx = None;
            self.ai_is_thinking = false;
            if let Some(color) = self.ai_thinking_color.take() {
                match color {
                    ChessColor::White => self.ai_white = Some(agent),
                    ChessColor::Black => self.ai_black = Some(agent),
                }
            }
            self.apply_player_move(mv);
        }
    }

    fn update_status(&mut self) {
        let side = match self.game.side_to_move() {
            ChessColor::White => "White",
            ChessColor::Black => "Black",
        };
        self.status = if self.game.position.is_check() {
            format!("Check! {side} to move")
        } else {
            format!("{side} to move")
        };
    }

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let buf = frame.buffer_mut();

        match &self.screen {
            Screen::Menu => {
                self.menu.clone().render(area, buf);
            }
            Screen::Game => {
                let ui = GameUiState {
                    game_state: &self.game,
                    board_state: &self.board_state,
                    white_name: &self.white_name,
                    black_name: &self.black_name,
                    status: &self.status,
                };
                render_game_ui(&ui, area, buf);
            }
            Screen::GameOver { message } => {
                let ui = GameUiState {
                    game_state: &self.game,
                    board_state: &self.board_state,
                    white_name: &self.white_name,
                    black_name: &self.black_name,
                    status: &self.status,
                };
                render_game_ui(&ui, area, buf);

                // Centered overlay
                let overlay_w = 40u16.min(area.width);
                let overlay_h = 5u16.min(area.height);
                let [_, center_v, _] = Layout::vertical([
                    Constraint::Fill(1),
                    Constraint::Length(overlay_h),
                    Constraint::Fill(1),
                ])
                .areas(area);
                let [_, overlay_area, _] = Layout::horizontal([
                    Constraint::Fill(1),
                    Constraint::Length(overlay_w),
                    Constraint::Fill(1),
                ])
                .areas(center_v);

                Clear.render(overlay_area, buf);
                let text = Text::from(vec![
                    Line::from(message.as_str()),
                    Line::from(""),
                    Line::from("Press any key to continue"),
                ]);
                Paragraph::new(text)
                    .style(Style::new().fg(Color::White).bg(Color::DarkGray))
                    .block(Block::bordered().title("Game Over"))
                    .render(overlay_area, buf);
            }
        }
    }
}
