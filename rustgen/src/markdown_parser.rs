
use crate::syntax_highlight::SyntaxHighlighter;

struct MarkdownParser<'a> {
    sh: Option<&'a SyntaxHighlighter>
}

#[derive(Debug)]
enum Tok {
    Metadata(Vec<(String,String)>), // key-value pairs
    Section(u8, String),
    Ticked(String),
    TripleTicked(String,String), // (label, code)
    Underscored(String),
    OneStarred(String),
    TwoStarred(String),
    Plaintext(String),
    Link(String, String), // (label, src)
    Image(String, String), // (label, src)
}

//
// str slices work by using byte offsets, but asserting that 
// any slice lies on a utf8-char boundary.
//
struct Stream<'a> {
    s: &'a str,
    i: usize,
}
// Lifetime 'b should be contained within 'a
struct StreamUse<'a,'b> {
    strm: &'b mut Stream<'a>,
    j: usize,
    is_beginning_of_line_: bool,
}
impl<'a,'b> StreamUse<'a,'b> {
    fn from(strm: &'b mut Stream<'a>) -> StreamUse<'a,'b> {
        let j = strm.i;
        let is_beginning_of_line_ = (j == 0) || (strm.s[j..j+1].chars().next().unwrap_or('.') == '\n');
        return StreamUse{strm:strm, j, is_beginning_of_line_};
    }

    fn commit(&mut self) -> () {
        self.strm.i = self.j;
    }

    fn is_beginning_of_line(&self) -> bool {
        return self.is_beginning_of_line_;
    }

    fn read_until_space(&mut self) -> Option<&str> {
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c!=' ' && c!='\t' && c!='\n' }).count();
        println!(" - read until space : {} -> {}", self.j, self.j+n);
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
    fn read_space(&mut self) -> Option<&str> {
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c==' ' || c=='\t' || c=='\n' }).count();
        println!(" - read space : {} -> {}", self.j, self.j+n);
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
    fn read_until_end_of_line(&mut self) -> Option<&str> {
        println!(" - read until eol : {}", self.j);
        let oj = self.j;
        let n = self.strm.s[self.j..].chars().take_while(|&c| { c!='\n' }).count();
        self.j += n;
        return Some(&self.strm.s[oj..self.j]);
    }
}
impl<'a,'b> Drop for StreamUse<'a,'b> {
    fn drop(&mut self) -> () { }
}

fn try_section(strm: &mut Stream) -> Option<Tok> {
    let mut user = StreamUse::from(strm);
    let start_j = user.j;

    if !user.is_beginning_of_line() { return None }

    // user.read_until_space();
    // user.read_space();
    if let Some(txt) = user.read_until_space() {
        if !(*txt).chars().all(|c| c=='#') {
            return None;
        }
        let n = txt.len();
        let title = user.read_until_end_of_line().unwrap().to_owned();

        user.commit();
        println!("emit section {} -> {}", start_j, user.j);
        return Some(Tok::Section(n as u8, title));
    }

    None
}

impl<'a> MarkdownParser<'a> {

    fn lex(&self, txt: &str) -> Vec<Tok> {
        use Tok::*;
        // let out = vec![Ticked(String::from("hello"))];
        let mut out = vec![];

        // Line-based
        /*
        let lines = txt.split("\n");
        for line in lines {
            if let Some(c) = line.chars().next() {
                if c == '#' {
                    let n = line.chars().take_while(|c| { *c == '#'}).count();
                    out.push(Section(n as u8, line.chars().skip(n).collect::<String>()));
                }
            }
        }
        */

        // More traditional + general scanner based
        let mut strm = Stream{s:txt, i:0};
        loop {
            if let Some(tok) = try_section(&mut strm) {
                out.push(tok);
            }

            break;
        }

        return out;
    }

    fn parse(&self, txt: &str) -> String {
        let out = String::new();

        let toks = self.lex(txt);

        return out;
    }

}

#[test]
fn test_md() {
    let src = r#"# SectionA
## SectionB
hello.
"#;

    let mdp = MarkdownParser {sh: None};
    let toks = mdp.lex(&src);
    toks.into_iter().map(|t| println!("{:?}",t)).for_each(drop);

    let s = "hello world!!";
    let mut strm = Stream{s:s, i:0};
    {
        let u1 = StreamUse::from(&mut strm);
    }
    {
        let u2 = StreamUse::from(&mut strm);
    }

}
