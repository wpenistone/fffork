use std::fmt;
use std::mem;
use std::ops::{Deref, Index};

use crate::glob_detect::has_wildcards;

/// Constraint types that can be extracted from a query
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint<'a> {
    /// Match file extension: *.rs -> Extension("rs")
    Extension(&'a str),

    /// Glob pattern: **/*.rs -> Glob("**/*.rs")
    Glob(&'a str),

    /// Multiple text search parts: ["src", "name"]
    /// Uses slice to avoid allocation
    Parts(&'a [&'a str]),

    /// Single text token (optimized case)
    Text(&'a str),

    /// Exclude pattern: !test -> Exclude(&["test"])
    Exclude(&'a [&'a str]),

    /// Path constraint: /src/ -> PathSegment("src")
    PathSegment(&'a str),

    /// File path constraint (AI mode): "libswscale/input.c" → FilePath("libswscale/input.c")
    /// Matches files whose relative path ends with this suffix at a `/` boundary.
    FilePath(&'a str),

    /// File type constraint: type:rust -> FileType("rust")
    FileType(&'a str),

    /// Git status constraint: status:modified -> GitStatus(Modified)
    GitStatus(GitStatusFilter),

    /// `!*.rs` is Not(Extension). `!*pat*` is Not(Glob). `!test` is Not(Text).
    Not(Box<Constraint<'a>>),
}

impl Constraint<'_> {
    /// Last component has a 1-10 byte extension starting with an ASCII letter.
    /// A trailing `/` rejects unless the token starts with `.`. Wildcards reject.
    #[inline(always)]
    pub fn is_filename_constraint_token(token: &str) -> bool {
        let bytes = token.as_bytes();
        if token.is_empty() || (bytes.last() == Some(&b'/') && bytes.first() != Some(&b'.')) {
            return false;
        }

        let filename = token.rsplit('/').next().unwrap_or(token);
        let Some(dot_idx) = filename.rfind('.') else {
            return false;
        };
        let extension = &filename[dot_idx + 1..];
        if !(extension.starts_with(|c: char| c.is_ascii_alphabetic())
            && extension.len() <= 10
            && extension.bytes().all(|b| b.is_ascii_alphanumeric()))
        {
            return false;
        }
        !has_wildcards(token)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitStatusFilter {
    Modified,
    Untracked,
    Staged,
    Unmodified,
}

/// Buffer for text parts during query parsing.
pub(crate) type TextPartsBuffer<'a> = Vec<&'a str>;

// The first constraint stays inline. A heap `Vec` starts at the second.
#[derive(Clone)]
pub struct ConstraintVec<'a> {
    rep: Rep<'a>,
}

#[derive(Clone)]
enum Rep<'a> {
    Empty,
    One(Constraint<'a>),
    Many(Vec<Constraint<'a>>),
}

impl<'a> ConstraintVec<'a> {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self { rep: Rep::Empty }
    }

    #[inline(always)]
    pub const fn new() -> Self {
        Self::empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.rep {
            Rep::Empty => 0,
            Rep::One(_) => 1,
            Rep::Many(rest) => rest.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(&self.rep, Rep::Empty)
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Constraint<'a>> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn as_slice(&self) -> &[Constraint<'a>] {
        match &self.rep {
            Rep::Empty => &[],
            Rep::One(one) => std::slice::from_ref(one),
            Rep::Many(rest) => rest.as_slice(),
        }
    }

    #[inline]
    pub fn push(&mut self, constraint: Constraint<'a>) {
        if let Rep::Many(rest) = &mut self.rep {
            rest.push(constraint);
            return;
        }
        if matches!(&self.rep, Rep::Empty) {
            self.rep = Rep::One(constraint);
            return;
        }
        let mut rest = Vec::with_capacity(4);
        let Rep::One(first) = mem::replace(&mut self.rep, Rep::Empty) else {
            unreachable!()
        };
        rest.push(first);
        rest.push(constraint);
        self.rep = Rep::Many(rest);
    }

    pub fn remove(&mut self, index: usize) -> Constraint<'a> {
        match mem::replace(&mut self.rep, Rep::Empty) {
            Rep::Empty => panic!("removal index (is {index}) should be < len (is 0)"),
            Rep::One(only) => {
                if index != 0 {
                    self.rep = Rep::One(only);
                    panic!("removal index (is {index}) should be < len (is 1)");
                }
                only
            }
            Rep::Many(mut rest) => {
                if index >= rest.len() {
                    let len = rest.len();
                    self.rep = Rep::Many(rest);
                    panic!("removal index (is {index}) should be < len (is {len})");
                }
                let removed = rest.remove(index);
                self.rep = match rest.len() {
                    0 => Rep::Empty,
                    1 => Rep::One(rest.pop().unwrap()),
                    _ => Rep::Many(rest),
                };
                removed
            }
        }
    }
}

impl Default for ConstraintVec<'_> {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> Deref for ConstraintVec<'a> {
    type Target = [Constraint<'a>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> Index<usize> for ConstraintVec<'a> {
    type Output = Constraint<'a>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a> Index<std::ops::RangeFull> for ConstraintVec<'a> {
    type Output = [Constraint<'a>];

    #[inline]
    fn index(&self, _: std::ops::RangeFull) -> &Self::Output {
        self.as_slice()
    }
}

impl<'a> AsRef<[Constraint<'a>]> for ConstraintVec<'a> {
    #[inline]
    fn as_ref(&self) -> &[Constraint<'a>] {
        self.as_slice()
    }
}

impl PartialEq for ConstraintVec<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl fmt::Debug for ConstraintVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<'a> FromIterator<Constraint<'a>> for ConstraintVec<'a> {
    fn from_iter<I: IntoIterator<Item = Constraint<'a>>>(iter: I) -> Self {
        let mut out = Self::empty();
        for constraint in iter {
            out.push(constraint);
        }
        out
    }
}

impl<'a, 'b> IntoIterator for &'b ConstraintVec<'a> {
    type Item = &'b Constraint<'a>;
    type IntoIter = std::slice::Iter<'b, Constraint<'a>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}
