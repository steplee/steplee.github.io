import React from 'react'
import Link from 'gatsby-link';
import Helmet from 'react-helmet';

import './common.css'

export default function CommonPage (content) {
  var header_style = {backgroundImage:'url(/hex0.png)',
                      width:'100%',
                      height:'520px',
                      backgroundRepeat:'no-repeat'
                     };
  return (
    <div className="my_container">

      <Helmet link={[
        {rel:'stylesheet', href:'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.3/css/bootstrap.min.css'},
        {rel:'stylesheet', href:'https://cdnjs.cloudflare.com/ajax/libs/prism/1.9.0/themes/prism-okaidia.min.css'}
      ]}
              script={[{'src':'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.3/js/bootstrap.min.js'}]}/>

      <Link to="https://fonts.googleapis.com/css?family=Cairo" rel="stylesheet"/>
      <Link to="https://fonts.googleapis.com/css?family=Roboto" rel="stylesheet"/>

      {/* HEADER */}
      <nav className="my_nav navbar navbar-expand-lg navbar-dark fixed-top">
        <div className="container">
          <a className="navbar-brand" href="#">steplee</a>
          <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarResponsive">
            <ul className="navbar-nav ml-auto">
              <li className="nav-item active">
                <a className="nav-link" href="https://steplee.github.io/">HOME
                  <span className="sr-only">(current)</span>
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#">POSTS</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="https://github.com/steplee">CODE</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="https://github.com/steplee/notes">NOTES</a>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      <div className="header_bg"></div>

      {/* CONTENT */}
      <div className="center-block common_content container">
        {content.children}
      </div>

      {/* FOOTER */}

    </div>
    );
}
