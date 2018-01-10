import React from 'react';
import Helmet from 'react-helmet';
import CommonPage from './common.js';

import './blog-post.css'

// https://www.gatsbyjs.org/blog/2017-07-19-creating-a-blog-with-gatsby/

export default function Template({ data })
{
  const { markdownRemark: post } = data; // injected by GraphQL query.
  return (
      <CommonPage>

          <Helmet title={`${post.frontmatter.title} - steplee blog`} />

          <section className="blog-post">
            <h1 className="post-title">{post.frontmatter.title}</h1>
            <hr/>
            <div className="blog-post-content"
                 dangerouslySetInnerHTML={{__html: post.html}}
            />
          </section>

      </CommonPage>
      );
}


export const pageQuery = graphql`
  query BlogPostByPath($path: String!) {
    markdownRemark(frontmatter: { path: {eq : $path} }) {
      html
      frontmatter {
        path
        title
      }
    }
  }`;
