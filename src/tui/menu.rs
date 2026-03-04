#![allow(dead_code, unused_imports)]

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{List, ListItem, Widget};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuSelection {
    #[default]
    HumanVsAI,
    HumanVsHuman,
    AIVsAI,
    Train,
    Quit,
}

impl MenuSelection {
    pub const ALL: &'static [Self] = &[
        Self::HumanVsAI,
        Self::HumanVsHuman,
        Self::AIVsAI,
        Self::Train,
        Self::Quit,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::HumanVsAI => "Human vs AI",
            Self::HumanVsHuman => "Human vs Human",
            Self::AIVsAI => "AI vs AI",
            Self::Train => "Train",
            Self::Quit => "Quit",
        }
    }
}

#[derive(Default)]
pub struct MainMenu {
    pub selected_index: usize,
}

impl MainMenu {
    pub fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    pub fn move_down(&mut self) {
        if self.selected_index < MenuSelection::ALL.len() - 1 {
            self.selected_index += 1;
        }
    }

    pub fn current(&self) -> MenuSelection {
        MenuSelection::ALL[self.selected_index]
    }
}

impl Widget for MainMenu {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let n_items = MenuSelection::ALL.len() as u16;

        // Center vertically
        let [_, content_v, _] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(n_items),
            Constraint::Fill(1),
        ])
        .areas(area);

        // Center horizontally
        let [_, content_h, _] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length(20),
            Constraint::Fill(1),
        ])
        .areas(content_v);

        let items: Vec<ListItem> = MenuSelection::ALL
            .iter()
            .enumerate()
            .map(|(i, &sel)| {
                let style = if i == self.selected_index {
                    Style::new()
                        .fg(Color::Black)
                        .bg(Color::White)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::new().fg(Color::DarkGray)
                };
                ListItem::new(sel.label()).style(style)
            })
            .collect();

        List::new(items).render(content_h, buf);
    }
}
