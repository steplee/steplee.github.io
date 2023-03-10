
//
// Crate to make the index page(s)
//

pub struct BlogMetadata {
    pub title: Option<String>,
    pub tags: Option<Vec<String>>,
    pub publish_date: Option<String>,
    pub modify_date: Option<String>
}
pub struct BlogPost {
    pub md_path: String,
    pub html_path: String,
    pub local_html_path: String,
    pub meta: BlogMetadata
}
pub struct BlogIndex {
    pub posts: Vec<BlogPost>,
}

impl BlogMetadata {
    fn new() -> BlogMetadata {
        return BlogMetadata{title: None, tags: None, publish_date: None, modify_date: None};
    }
}

pub fn find_posts(md_path: &str, out_path: &str, local_path: &str) -> BlogIndex {
    use std::fs;
    use std::path::Path;
    use std::ffi::OsStr;
    let mut posts = vec![];
    for entry in fs::read_dir(md_path).expect(&format!("failed to read_dir '{}'", &md_path)) {

        if let Ok(entry) = entry {
            let fname_os = entry.file_name();
            let fname = fname_os.to_str().unwrap();
            if fname.ends_with(".md") || fname.ends_with(".MD") {

                let post_title = fname.rsplit("/").next().unwrap();
                let post_title = post_title.split(".").next().unwrap();
                let post_path = format!("{out_path}/{post_title}.html");
                let post_path_local = format!("{local_path}/{post_title}.html");

                let post = BlogPost {
                    md_path: entry.path().display().to_string(),
                    html_path: post_path,
                    local_html_path: post_path_local,
                    meta: BlogMetadata::new()
                };

                // posts.push(entry.path().to_str().unwrap().to_owned());
                posts.push(post);
            }
        }
    }

    let index = BlogIndex { posts: posts };

    // println!("posts {:?}", index);

    return index;
}

