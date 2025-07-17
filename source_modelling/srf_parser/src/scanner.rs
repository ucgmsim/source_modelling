use lexical_core::Error::*;

use std::error;
use std::fmt;
use std::io::Write;

#[derive(Debug)]
pub struct ScannerError {
    context: String,
    error: Box<dyn error::Error>,
}

impl ScannerError {
    pub fn new(data: &[u8], error: impl error::Error + 'static) -> Self {
        Self {
            context: String::from_utf8_lossy(data).into_owned(),
            error: Box::new(error),
        }
    }
}

impl fmt::Display for ScannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, context: {}", self.error, self.context)
    }
}

impl error::Error for ScannerError {}

#[derive(Debug)]
enum ScannerErrorReason {
    InvalidToken(String, String),
    NoNewlineFound,
}

impl fmt::Display for ScannerErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken(expected, found) => {
                write!(f, "Invalid token, expected: {}, found: {}", expected, found)
            }
            Self::NoNewlineFound => write!(f, "Could not find newline"),
        }
    }
}

impl error::Error for ScannerErrorReason {}

pub struct Scanner<'a> {
    data: &'a [u8],
    pub index: usize,
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, index: 0 }
    }

    pub fn next<T: lexical_core::FromLexical>(&mut self) -> Result<T, ScannerError> {
        self.skip_spaces()?;
        let (val, read) = lexical_core::parse_partial(&self.data[self.index..])
            .map_err(|err| match err {
                Overflow(offset) => Overflow(self.index + offset),
                Underflow(offset) => Underflow(self.index + offset),
                InvalidDigit(offset) => InvalidDigit(self.index + offset),
                Empty(offset) => Empty(self.index + offset),
                EmptyMantissa(offset) => EmptyMantissa(self.index + offset),
                EmptyExponent(offset) => EmptyExponent(self.index + offset),
                EmptyInteger(offset) => EmptyInteger(self.index + offset),
                EmptyFraction(offset) => EmptyFraction(self.index + offset),
                InvalidPositiveMantissaSign(offset) => {
                    InvalidPositiveMantissaSign(self.index + offset)
                }
                MissingMantissaSign(offset) => MissingMantissaSign(self.index + offset),
                InvalidExponent(offset) => InvalidExponent(self.index + offset),
                InvalidPositiveExponentSign(offset) => {
                    InvalidPositiveExponentSign(self.index + offset)
                }
                MissingExponentSign(offset) => MissingExponentSign(self.index + offset),
                ExponentWithoutFraction(offset) => ExponentWithoutFraction(self.index + offset),
                InvalidLeadingZeros(offset) => InvalidLeadingZeros(self.index + offset),
                MissingExponent(offset) => MissingExponent(self.index + offset),
                MissingSign(offset) => MissingSign(self.index + offset),
                InvalidPositiveSign(offset) => InvalidPositiveSign(self.index + offset),
                InvalidNegativeSign(offset) => InvalidNegativeSign(self.index + offset),
                e => e,
            })
            .map_err(|err| ScannerError::new(self.context(), err))?;
        self.index += read;
        Ok(val)
    }

    fn skip_spaces(&mut self) -> Result<(), ScannerError> {
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
            _ => Err(ScannerError::new(self.context(), InvalidDigit(self.index))),
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
            _ => Err(ScannerError::new(
                self.context(),
                ScannerErrorReason::NoNewlineFound,
            )),
        }
    }

    pub fn skip_token(&mut self, token: &[u8]) -> Result<(), ScannerError> {
        self.skip_spaces()?;
        let next = &self.data[self.index..self.index + token.len()];
        if next == token {
            self.index += token.len();
            Ok(())
        } else {
            Err(ScannerError::new(
                self.context(),
                ScannerErrorReason::InvalidToken(
                    String::from_utf8_lossy(token).into_owned(),
                    String::from_utf8_lossy(next).into_owned(),
                ),
            ))
        }
    }

    pub fn context(&self) -> &[u8] {
        return &self.data[self.index..(self.index + 20).min(self.data.len())];
    }
}
