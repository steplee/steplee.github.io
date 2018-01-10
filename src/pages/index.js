import React from 'react';
import Link from 'gatsby-link';
import Helmet from 'react-helmet';
import CommonPage from '../templates/common';

import './index.css';

export default function Index({ data }) {
  const { edges: posts } = data.allMarkdownRemark;
  return (
      <CommonPage>
        <section className="span12">
          {posts.filter(post => post.node.frontmatter.title.length > 0)
                .map(({node: post}) => {
                  return (
                      <div className="blog-post-preview" key={post.id}>
                        <h1> <Link to={post.frontmatter.path}>
                          {post.frontmatter.title}
                        </Link> </h1>
                        <h2> {post.frontmatter.date} </h2>
                        <p>  {post.excerpt} </p>
                      </div>
                      );
                  })}
          </section>
      </CommonPage>
      );
}

export const pageQuery = graphql`
  query IndexQuery {
    allMarkdownRemark(sort: { order: DESC, fields: [frontmatter___date] }) {
      edges {
        node {
          excerpt(pruneLength: 250)
          id
          frontmatter {
            title
            date(formatString: "MMMM DD, YY")
            path
          }
        }
      }
    }
  }`
