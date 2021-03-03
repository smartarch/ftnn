name := "trust4.0-demo-enforcer"

version := "1.0"

scalaVersion := "2.12.8"

mainClass in (Compile, run) := Some("trust40.Main")

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.12.8",
  "org.scala-lang.modules" %% "scala-xml" % "1.2.0",
  "org.choco-solver" % "choco-solver" % "4.10.0",
  "com.typesafe.akka" %% "akka-actor" % "2.5.22",
  "com.typesafe.akka" %% "akka-stream" % "2.5.22",
  "com.typesafe.akka" %% "akka-http" % "10.1.8",
  "com.typesafe.akka" %% "akka-http-spray-json" % "10.1.8",
  "org.junit.jupiter" % "junit-jupiter-api" % "5.5.1",
  "org.junit.jupiter" % "junit-jupiter-engine" % "5.5.1",
  "org.hamcrest" % "hamcrest-all" % "1.3"
)
