package ftnn

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import ftnn.k4case.{Position, Simulation, SimulationState, WorkerState}
import spray.json.{DefaultJsonProtocol, JsNumber, JsString, JsValue, JsonFormat, deserializationError}

import java.time.format.DateTimeFormatter
import java.time.{LocalDateTime, ZoneOffset}

trait MarshallersSupport extends SprayJsonSupport with DefaultJsonProtocol {
  implicit object SimulationStateStateJsonFormat extends JsonFormat[Simulation.State.State] {
    def write(x: Simulation.State.State) = JsNumber(x.id)
    def read(value: JsValue) = value match {
      case JsNumber(x) => Simulation.State(x.toInt)
      case x => deserializationError("Expected Int as JsNumber, but got " + x)
    }
  }

  implicit object LocalDateTimeJsonFormat extends JsonFormat[LocalDateTime] {
    def write(x: LocalDateTime) = JsString(x.atZone(ZoneOffset.UTC).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME))
    def read(value: JsValue) = value match {
      case JsString(x) => LocalDateTime.parse(x, DateTimeFormatter.ISO_OFFSET_DATE_TIME)
      case x => deserializationError("Expected Int as JsNumber, but got " + x)
    }
  }

  implicit val positionFormat = jsonFormat2(Position)
  implicit val workerStateFormat = jsonFormat3(WorkerState)
  implicit val simulationStateFormat = jsonFormat4(SimulationState)
}
