from annotationapp.app import db
from sqlalchemy.dialects import mysql


class TRACK(db.Model):

    __tablename__ = 'TRACK'
    
    id = db.Column(db.BIGINT, primary_key=True)
    external_id = db.Column(db.String(length=50))
    datetime=db.Column(db.DateTime)
    component_length=db.Column(db.Integer)
    def __repr__(self):
        return "<TRACK(id='{}',datetime ='{}',component_length='{}')>".format(self.id,
                           self.datetime,self.component_length)


class TRACKPOINT(db.Model):
    __tablename__ = 'TRACKPOINT'

    id = db.Column(db.BIGINT, primary_key=True)
    track_id=db.Column(db.BIGINT,db.ForeignKey('TRACK.id'))

    latitude=db.Column(mysql.DOUBLE())
    longitude=db.Column(mysql.DOUBLE())
    altitude=db.Column(mysql.DOUBLE())
    speed = db.Column(mysql.DOUBLE())
    bearing = db.Column(mysql.DOUBLE())
    datetime=db.Column(db.DateTime)
    millisecond=db.Column(db.Integer)


    def __repr__(self):
        return "<TRACKPOINT(track_id='{0}', latitude='{1}', longitude='{2}',altitude='{3}',datetime='{4}',millisecond='{5}')>".format(
                            self.track_id, self.latitude, self.longitude,self.altitude,self.datetime,self.millisecond)



db.create_all()