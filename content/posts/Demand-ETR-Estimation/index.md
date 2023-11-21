---
title: "Demand and ETR Estimation"
date: 2023-10-25T13:33:46+02:00
draft: true

---



Introduction



Airports currently hold a significant portion of Uber’s supply and open supply hours (i.e., supply that is not utilized, but open for dispatch) across the globe. At most airports, drivers are obligated to join a “first-in-first-out” (FIFO) queue from which they are dispatched. When the demand for trips is high relative to the supply of drivers in the queue (“undersupply”), this queue moves quickly and wait times for drivers can be quite low. However, when demand is low relative to the amount of available supply (“oversupply”), the queue moves slowly and wait times can be very high. Undersupply creates a poor experience for riders, as they are less likely to get a suitable ride. On the other hand, oversupply creates a poor experience for drivers as they are spending more time waiting for each ride and less time driving. What’s more, drivers don’t currently have a way to see when airports are under- or over-supplied, which perpetuates this problem.

One way to tackle this undersupply/oversupply issue at airports is to forecast supply balance and use this to optimize resource allocation. Our first application of these models is in estimating the time to request (ETR) for the airport driver queue. We estimate the length of time a driver would have to wait before they receive a trip request, thereby giving drivers the information they need to identify and reposition in periods of undersupply (short waits), or to remain in the city during periods of oversupply (long waits). 



