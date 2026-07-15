use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScannerError {
    #[error("invalid number at byte {index}: {source}, context: {context}")]
    InvalidNumber {
        source: lexical_core::Error,
        index: usize,
        context: String,
    },
    #[error("invalid token, expected: {expected}, found: {found}")]
    InvalidToken { expected: String, found: String },
    #[error("could not find newline, context: {context}")]
    NoNewlineFound { context: String },
    #[error("unexpected end of input, context: {context}")]
    UnexpectedEof { context: String },
}

pub struct Scanner<'a> {
    data: &'a [u8],
    pub index: usize,
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, index: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.index
    }

    pub fn peek(&self) -> u8 {
        self.data[self.index]
    }

    pub fn next<T: lexical_core::FromLexical>(&mut self) -> Result<T, ScannerError> {
        self.skip_spaces()?;
        let (val, read) =
            lexical_core::parse_partial(&self.data[self.index..]).map_err(|source| {
                ScannerError::InvalidNumber {
                    source,
                    index: self.index,
                    context: self.context_string(),
                }
            })?;
        self.index += read;
        Ok(val)
    }

    pub fn skip_spaces(&mut self) -> Result<(), ScannerError> {
        let nonwhitespace = &self.data[self.index..]
            .iter()
            .enumerate()
            .find(|&(_, &x)| !x.is_ascii_whitespace())
            .map(|(idx, _)| idx);
        match nonwhitespace {
            Some(x) => {
                self.index += x;
                Ok(())
            }
            _ => Err(ScannerError::UnexpectedEof {
                context: self.context_string(),
            }),
        }
    }

    pub fn expect_end_of_line(&mut self) -> Result<(), ScannerError> {
        let mut i = self.index;
        let mut found_eol = false;

        while i < self.data.len() && !found_eol {
            let c = self.data[i];
            if c == b'\n' {
                found_eol = true;
            } else if !c.is_ascii_whitespace() {
                break;
            }
            i += 1;
        }

        if found_eol {
            self.index = i;
            Ok(())
        } else if i == self.data.len() {
            Err(ScannerError::UnexpectedEof {
                context: self.context_string(),
            })
        } else {
            Err(ScannerError::NoNewlineFound {
                context: self.context_string(),
            })
        }
    }

    pub fn line(&mut self) -> Result<&[u8], ScannerError> {
        let newline_index = &self.data[self.index..]
            .iter()
            .enumerate()
            .find(|&(_, &x)| x == b'\n')
            .map(|(idx, _)| idx);
        match newline_index {
            Some(x) => {
                let res = Ok(&self.data[self.index..self.index + x]);
                self.index += x + 1; // plus 1 to skip the newline itself.
                res
            }
            _ => Err(ScannerError::NoNewlineFound {
                context: self.context_string(),
            }),
        }
    }

    pub fn skip_token(&mut self, token: &[u8]) -> Result<(), ScannerError> {
        self.skip_spaces()?;
        let next = self
            .data
            .get(self.index..self.index + token.len())
            .ok_or_else(|| ScannerError::UnexpectedEof {
                context: self.context_string(),
            })?;
        if next == token {
            self.index += token.len();
            Ok(())
        } else {
            Err(ScannerError::InvalidToken {
                expected: String::from_utf8_lossy(token).into_owned(),
                found: String::from_utf8_lossy(next).into_owned(),
            })
        }
    }

    pub fn context(&self) -> &[u8] {
        &self.data[self.index..(self.index + 20).min(self.data.len())]
    }

    fn context_string(&self) -> String {
        String::from_utf8_lossy(self.context()).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_parses_numbers_across_whitespace() {
        let mut scanner = Scanner::new(b"1.5  -2 \n 3e2");
        assert_eq!(scanner.next::<f32>().unwrap(), 1.5);
        assert_eq!(scanner.next::<i32>().unwrap(), -2);
        assert_eq!(scanner.next::<f32>().unwrap(), 300.0);
    }

    #[test]
    fn next_error_carries_position_and_context() {
        let mut scanner = Scanner::new(b"1.0 abc");
        scanner.next::<f32>().unwrap();
        let err = scanner.next::<f32>().unwrap_err();
        match err {
            ScannerError::InvalidNumber { index, context, .. } => {
                assert_eq!(index, 4);
                assert_eq!(context, "abc");
            }
            other => panic!("expected InvalidNumber, got {other:?}"),
        }
    }

    #[test]
    fn next_on_exhausted_input_is_eof() {
        let mut scanner = Scanner::new(b"  \n ");
        assert!(matches!(
            scanner.next::<f32>().unwrap_err(),
            ScannerError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn skip_token_matches_and_advances() {
        let mut scanner = Scanner::new(b"  POINTS 2");
        scanner.skip_token(b"POINTS").unwrap();
        assert_eq!(scanner.next::<usize>().unwrap(), 2);
    }

    #[test]
    fn skip_token_mismatch_reports_both_tokens() {
        let mut scanner = Scanner::new(b"PLANES");
        match scanner.skip_token(b"POINTS").unwrap_err() {
            ScannerError::InvalidToken { expected, found } => {
                assert_eq!(expected, "POINTS");
                assert_eq!(found, "PLANES");
            }
            other => panic!("expected InvalidToken, got {other:?}"),
        }
    }

    #[test]
    fn skip_token_on_truncated_input_errors_without_panicking() {
        let mut scanner = Scanner::new(b"POIN");
        assert!(matches!(
            scanner.skip_token(b"POINTS").unwrap_err(),
            ScannerError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn line_reads_until_newline() {
        let mut scanner = Scanner::new(b"1.0\nrest");
        assert_eq!(scanner.line().unwrap(), b"1.0");
        assert_eq!(scanner.remaining(), 4);
    }

    #[test]
    fn line_without_newline_errors() {
        let mut scanner = Scanner::new(b"no newline here");
        assert!(matches!(
            scanner.line().unwrap_err(),
            ScannerError::NoNewlineFound { .. }
        ));
    }
}
